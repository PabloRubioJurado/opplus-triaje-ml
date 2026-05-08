import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Configuración profesional
st.set_page_config(page_title="OPPLUS - Triaje Inteligente", layout="wide")
st.title("Sistema de Triaje GYAR - Equipo Houston")

# --- PERSISTENCIA DE DATOS EN LA SESIÓN ---
if 'df_operativo' not in st.session_state:
    st.session_state.df_operativo = None

# --- 1. CARGA DE EXPEDIENTES ---
st.markdown("### 1. Carga de Expedientes Operativos")
archivo_subido = st.file_uploader("Suba el CSV con la base de datos completa", type=["csv"])

if archivo_subido is not None and st.session_state.df_operativo is None:
    df_inicial = pd.read_csv(archivo_subido)
    
    # Inicializamos columnas de control si no existen
    if 'Gestionado_Hoy' not in df_inicial.columns:
        df_inicial['Gestionado_Hoy'] = False  # Para quitar de la lista definitiva
    if 'Llamadas_Previas' not in df_inicial.columns:
        df_inicial['Llamadas_Previas'] = 0
        
    st.session_state.df_operativo = df_inicial

# --- 2. MOTOR DE IA (RE-TRIADORA) ---
def ejecutar_ia_triaje(df):
    """Entrena y predice el riesgo basado en el estado actual de los datos"""
    columnas_ia = ['Importe_Deuda', 'Dias_Impago', 'Prioridad_Banco', 'Llamadas_Previas']
    
    # Simulación de entrenamiento (solo si no viene con etiquetas de mora)
    df_entreno = df.copy()
    if 'Llego_a_Mora' not in df_entreno.columns:
        prob = (df_entreno['Dias_Impago'] / 90) * 0.6 + \
               (df_entreno['Importe_Deuda'] / df_entreno['Importe_Deuda'].max()) * 0.3 - \
               (df_entreno['Llamadas_Previas'] * 0.1)
        df_entreno['Llego_a_Mora'] = (prob > 0.55).astype(int)
    
    modelo = DecisionTreeClassifier(max_depth=4)
    modelo.fit(df_entreno[columnas_ia], df_entreno['Llego_a_Mora'])
    
    # Generamos el Score de Urgencia actualizado
    df['Score_Urgencia'] = np.round(modelo.predict_proba(df[columnas_ia])[:, 1] * 100, 2)
    return df.sort_values(by='Score_Urgencia', ascending=False)

# --- 3. PANEL DE CONTROL Y GESTIÓN ---
if st.session_state.df_operativo is not None:
    # Ejecutamos la IA sobre toda la base de datos
    df_maestro = ejecutar_ia_triaje(st.session_state.df_operativo)
    
    # Filtramos para la vista: Solo los NO terminados
    df_solo_pendientes = df_maestro[df_maestro['Gestionado_Hoy'] == False]
    
    # Tomamos los 100 más críticos para mostrar
    df_vista_100 = df_solo_pendientes.head(100).copy()

    # --- MÉTRICAS GENERALES ---
    st.divider()
    total_datos = len(st.session_state.df_operativo)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Expedientes en Base", f"{total_datos:,}")
    
    pendientes_ahora = len(df_solo_pendientes)
    c2.metric("Pendientes de Gestión", f"{pendientes_ahora:,}", delta=f"-{total_datos - pendientes_ahora} hoy")
    
    capital_en_vuelo = df_vista_100['Importe_Deuda'].sum()
    c3.metric("Capital en Gestión Actual (Top 100)", f"{capital_en_vuelo:,.0f} €")

    # --- BANDEJA DE ENTRADA INTELIGENTE ---
    st.subheader(f"Bandeja Operativa: Mostrando los 100 casos con mayor prioridad")
    st.write(f"ℹ️ El motor de IA ha seleccionado estos 100 expedientes de un total de {pendientes_ahora:,} casos disponibles.")

    # Añadimos columnas de acción al editor
    df_vista_100['✅ Gestión Cerrada'] = False
    df_vista_100['📞 Llamada Fallida (No contesta)'] = False

    # El Editor de Datos
    df_editado = st.data_editor(
        df_vista_100[['✅ Gestión Cerrada', '📞 Llamada Fallida (No contesta)', 'ID_Cliente', 'Score_Urgencia', 'Llamadas_Previas', 'Importe_Deuda', 'Dias_Impago']],
        use_container_width=True,
        hide_index=True,
        disabled=['ID_Cliente', 'Score_Urgencia', 'Llamadas_Previas', 'Importe_Deuda', 'Dias_Impago']
    )

    # --- BOTÓN DE PROCESAMIENTO ---
    if st.button("🚀 Procesar Cambios y Actualizar los 100 mejores"):
        # Identificar quién se ha marcado
        finalizados = df_editado[df_editado['✅ Gestión Cerrada'] == True]['ID_Cliente'].values
        reintentos = df_editado[df_editado['📞 Llamada Fallida (No contesta)'] == True]['ID_Cliente'].values

        # Actualizar la sesión maestra
        for idx in st.session_state.df_operativo.index:
            cid = st.session_state.df_operativo.at[idx, 'ID_Cliente']
            
            if cid in finalizados:
                st.session_state.df_operativo.at[idx, 'Gestionado_Hoy'] = True
            
            if cid in reintentos:
                st.session_state.df_operativo.at[idx, 'Llamadas_Previas'] += 1
                # Al sumar una llamada, el Score de Urgencia cambiará en la siguiente vuelta

        st.success("Base de datos sincronizada. Trayendo nuevos casos a la bandeja...")
        st.rerun()

else:
    st.info("ℹ️ Cargue el archivo CSV para procesar el triaje inteligente de la jornada.")
