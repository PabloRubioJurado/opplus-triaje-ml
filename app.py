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

# --- LÓGICA DE DATOS (REAL O SIMULADA) ---
if archivo_subido is not None:
    # Caso A: El usuario sube su propio archivo
    df = pd.read_csv(archivo_subido)
    fuente_datos = "real"
else:
    # Caso B: Modo Simulación (Demo) para que la app no esté vacía
    st.info("Modo de simulación activado: Generando 12.000 expedientes para demostración del modelo...")
    np.random.seed(0)
    df = pd.DataFrame({
        'ID_Cliente': range(1, 12001),
        'Importe_Deuda': np.round(np.random.uniform(50, 15000, 12000), 2),
        'Dias_Impago': np.random.randint(1, 90, 12000),
        'Prioridad_Banco': np.random.choice([1, 2, 3], 12000, p=[0.6, 0.3, 0.1]),
        'Llamadas_Previas': np.random.randint(0, 6, 12000)
    })
    fuente_datos = "simulada"

# --- MOTOR DE INTELIGENCIA ARTIFICIAL ---
columnas_requeridas = ['Importe_Deuda', 'Dias_Impago', 'Prioridad_Banco', 'Llamadas_Previas']

# Verificamos que el archivo tenga lo necesario
if all(col in df.columns for col in columnas_requeridas):
    
    # Si los datos son nuevos o no traen la columna de 'objetivo', creamos una lógica de entrenamiento
    if 'Llego_a_Mora' not in df.columns:
        # Lógica matemática para que el árbol aprenda qué es "peligroso"
        prob = (df['Dias_Impago'] / 90) * 0.7 + (df['Importe_Deuda'] / df['Importe_Deuda'].max()) * 0.3
        df['Llego_a_Mora'] = (prob > 0.55).astype(int)

    # Entrenamiento del modelo
    X = df[columnas_requeridas]
    Y = df['Llego_a_Mora']
    modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
    modelo.fit(X, Y)
    
    # Cálculo de Score de Urgencia (Probabilidad de impago)
    df['Score_Urgencia'] = np.round(modelo.predict_proba(X)[:, 1] * 100, 2)
    df_final = df.sort_values(by='Score_Urgencia', ascending=False)

    if fuente_datos == "real":
        st.success(f"Análisis completado: {len(df)} expedientes procesados con éxito.")

    # --- DASHBOARD VISUAL ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    # Métrica 1: Volumen
    col1.metric("Expedientes a Gestionar", f"{len(df):,}", "Carga de trabajo")
    
    # Métrica 2: Casos Críticos
    casos_criticos = len(df_final[df_final['Score_Urgencia'] > 80])
    col2.metric("Casos Nivel Crítico (>80%)", f"{casos_criticos}", "Prioridad Máxima", delta_color="inverse")
    
    # Métrica 3: Capital en Riesgo
    dinero_riesgo = df_final.head(casos_criticos)['Importe_Deuda'].sum()
    col3.metric("Capital en Máximo Riesgo", f"{dinero_riesgo:,.0f} €", "Protección de Activos")

    # --- TABLA DE TRABAJO ---
    st.subheader("Bandeja de Trabajo Priorizada")
    st.markdown("La siguiente lista muestra los clientes ordenados por riesgo de impago calculado por la IA:")
    
    # Mostrar tabla con las columnas más relevantes
    st.dataframe(
        df_final[['ID_Cliente', 'Score_Urgencia', 'Dias_Impago', 'Importe_Deuda', 'Prioridad_Banco']], 
        use_container_width=True,
        hide_index=True
    )
    
    if st.button("Exportar lista de contacto a los gestores"):
        st.balloons()
        st.success("Lista enviada correctamente a los sistemas de centralita.")

else:
    st.error(f"Error: El archivo subido no contiene las columnas necesarias: {columnas_requeridas}")
