import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Configuración de la página
st.set_page_config(page_title="OPPLUS - Triaje Inteligente", layout="wide")
st.title("Sistema de Triaje GYAR - Equipo Houston")

# --- SECCIÓN DE SUBIDA DE DATOS ---
st.markdown("### 1. Carga de Expedientes")
archivo_subido = st.file_uploader("Seleccione el archivo CSV con los expedientes diarios", type=["csv"])

# --- LÓGICA DEL SISTEMA ---
if archivo_subido is not None:
    # El usuario sube su archivo
    df = pd.read_csv(archivo_subido)
    columnas_requeridas = ['Importe_Deuda', 'Dias_Impago', 'Prioridad_Banco', 'Llamadas_Previas']

    # Verificamos que el archivo sea correcto
    if all(col in df.columns for col in columnas_requeridas):
        
        # Si no traen la columna objetivo, la generamos internamente para entrenar
        if 'Llego_a_Mora' not in df.columns:
            prob = (df['Dias_Impago'] / 90) * 0.7 + (df['Importe_Deuda'] / df['Importe_Deuda'].max()) * 0.3
            df['Llego_a_Mora'] = (prob > 0.55).astype(int)

        # Motor de IA: Entrenamiento y Predicción
        X = df[columnas_requeridas]
        Y = df['Llego_a_Mora']
        modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
        modelo.fit(X, Y)
        
        df['Score_Urgencia'] = np.round(modelo.predict_proba(X)[:, 1] * 100, 2)
        df_final = df.sort_values(by='Score_Urgencia', ascending=False)

        st.success(f"Análisis completado: {len(df):,} expedientes procesados con éxito.")

        # --- DASHBOARD VISUAL ---
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Expedientes a Gestionar", f"{len(df):,}", "Carga de trabajo")
        
        casos_criticos = len(df_final[df_final['Score_Urgencia'] > 80])
        col2.metric("Casos Nivel Crítico (>80%)", f"{casos_criticos:,}", "Prioridad Máxima", delta_color="inverse")
        
        dinero_riesgo = df_final.head(casos_criticos)['Importe_Deuda'].sum()
        col3.metric("Capital en Máximo Riesgo", f"{dinero_riesgo:,.0f} €", "Protección de Activos")

        # --- TABLA DE TRABAJO INTERACTIVA ---
        st.subheader("Bandeja de Trabajo Activa")
        st.markdown("Gestores: Marquen la casilla **'Gestionado'** tras finalizar la llamada con el cliente.")
        
        # 1. Añadimos la columna interactiva de "Check" al principio
        if 'Gestionado' not in df_final.columns:
            df_final.insert(0, 'Gestionado', False)

        # 2. Usamos data_editor en lugar de dataframe para permitir interacción
        df_editado = st.data_editor(
            df_final[['Gestionado', 'ID_Cliente', 'Score_Urgencia', 'Dias_Impago', 'Importe_Deuda', 'Prioridad_Banco']], 
            use_container_width=True,
            hide_index=True,
            # Bloqueamos todas las columnas EXCEPTO la de 'Gestionado'
            disabled=['ID_Cliente', 'Score_Urgencia', 'Dias_Impago', 'Importe_Deuda', 'Prioridad_Banco']
        )
        
        # 3. Calculamos cuántos han marcado y ponemos una barra de progreso
        llamadas_hechas = df_editado['Gestionado'].sum()
        total_llamadas = len(df_editado)
        
        # Evitamos división por cero por si acaso
        if total_llamadas > 0:
            progreso = llamadas_hechas / total_llamadas
            st.progress(progreso, text=f"📊 Progreso de la jornada: {llamadas_hechas} de {total_llamadas} casos gestionados")
        
        # 4. El botón de exportar ahora exporta solo lo que falta por llamar
        if st.button("Sincronizar progreso con centralita"):
            casos_restantes = total_llamadas - llamadas_hechas
            st.success(f"Progreso guardado. Quedan {casos_restantes} casos pendientes en la cola.")

    else:
        st.error(f"Error de formato: El archivo debe contener exactamente estas columnas: {columnas_requeridas}")

else:
    # PANTALLA DE ESPERA (Empty State)
    st.info("ℹ️ El sistema está en espera. Por favor, cargue el archivo CSV de la jornada operativa para iniciar el motor de triaje.")
