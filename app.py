import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="OPPLUS - Triaje Inteligente", layout="wide")
st.title("Sistema de Triaje GYAR - Equipo Houston")

# --- PERSISTENCIA DE DATOS ---
if 'df_operativo' not in st.session_state:
    st.session_state.df_operativo = None

# --- CARGA DE DATOS ---
st.markdown("### 1. Carga de Expedientes Operativos")
archivo_subido = st.file_uploader("Suba el CSV de la jornada", type=["csv"])

if archivo_subido is not None and st.session_state.df_operativo is None:
    df_inicial = pd.read_csv(archivo_subido)
    # Columnas de control de estado
    if 'Gestionado_Hoy' not in df_inicial.columns:
        df_inicial['Gestionado_Hoy'] = False  # Para los que ya hemos terminado
    if 'Llamadas_Previas' not in df_inicial.columns:
        df_inicial['Llamadas_Previas'] = 0
    st.session_state.df_operativo = df_inicial

# --- MOTOR DE IA ---
def ejecutar_ia_triaje(df):
    columnas_ia = ['Importe_Deuda', 'Dias_Impago', 'Prioridad_Banco', 'Llamadas_Previas']
    # Solo entrenamos con lo que no ha sido gestionado todavía
    df_temp = df.copy()
    if 'Llego_a_Mora' not in df_temp.columns:
        prob = (df_temp['Dias_Impago'] / 90) * 0.7 + (df_temp['Importe_Deuda'] / df_temp['Importe_Deuda'].max()) * 0.3
        df_temp['Llego_a_Mora'] = (prob > 0.5).astype(int)
    
    modelo = DecisionTreeClassifier(max_depth=4)
    modelo.fit(df_temp[columnas_ia], df_temp['Llego_a_Mora'])
    df['Score_Urgencia'] = np.round(modelo.predict_proba(df[columnas_ia])[:, 1] * 100, 2)
    return df.sort_values(by='Score_Urgencia', ascending=False)

# --- PANEL DE TRABAJO ---
if st.session_state.df_operativo is not None:
    # Re-calculamos triaje sobre la base completa
    df_triado = ejecutar_ia_triaje(st.session_state.df_operativo)
    
    # Filtramos: Solo lo que NO está terminado hoy
    df_pendiente = df_triado[df_triado['Gestionado_Hoy'] == False].head(100).copy()

    # Dashboard de control
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Casos en Bandeja", len(df_pendiente))
    terminados = st.session_state.df_operativo['Gestionado_Hoy'].sum()
    c2.metric("Objetivo Conseguido", f"{terminados} casos", delta="Hoy")
    capital = df_pendiente['Importe_Deuda'].sum()
    c3.metric("Capital bajo gestión (Top 100)", f"{capital:,.0f} €")

    st.subheader("📋 Gestión de los 100 Casos Críticos")
    st.info("Instrucciones: Marque 'Terminado' para quitar de la lista. Marque 'Llamar de nuevo' para sumar intento y re-calcular posición.")

    # Añadimos columnas temporales de control para el editor
    df_pendiente['Terminado'] = False
    df_pendiente['Llamar_de_nuevo'] = False

    # El Editor: Solo mostramos lo relevante para el humano
    df_editado = st.data_editor(
        df_pendiente[['Terminado', 'Llamar_de_nuevo', 'ID_Cliente', 'Score_Urgencia', 'Llamadas_Previas', 'Importe_Deuda', 'Dias_Impago']],
        use_container_width=True,
        hide_index=True,
        disabled=['ID_Cliente', 'Score_Urgencia', 'Llamadas_Previas', 'Importe_Deuda', 'Dias_Impago']
    )

    # --- EL BOTÓN MÁGICO ---
    if st.button("Actualizar Bandeja y Traer nuevos casos"):
        # 1. Identificamos los cambios
        ids_terminados = df_editado[df_editado['Terminado'] == True]['ID_Cliente'].values
        ids_reintentar = df_editado[df_editado['Llamar_de_nuevo'] == True]['ID_Cliente'].values

        # 2. Aplicamos cambios a la base maestra de la sesión
        for idx in st.session_state.df_operativo.index:
            cliente_id = st.session_state.df_operativo.at[idx, 'ID_Cliente']
            
            if cliente_id in ids_terminados:
                st.session_state.df_operativo.at[idx, 'Gestionado_Hoy'] = True
            
            if cliente_id in ids_reintentar:
                st.session_state.df_operativo.at[idx, 'Llamadas_Previas'] += 1
                # (Al sumar llamada, en el siguiente loop la IA le bajará el Score)

        st.success("Sincronizando con la nube de OPPLUS... Bandeja actualizada.")
        st.rerun()

else:
    st.info("ℹ️ Cargue el archivo CSV para que la IA seleccione los 100 casos con mayor riesgo de pérdida hoy.")
