import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="OPPLUS - Triaje Inteligente", layout="wide")
st.title("Sistema de Triaje GYAR - Equipo Houston")

# --- SECCIÓN DE SUBIDA DE DATOS ---
st.markdown("### 1. Carga de Expedientes")
archivo_subido = st.file_uploader("Seleccione el archivo CSV con los expedientes diarios", type=["csv"])

# --- MOTOR DE INTELIGENCIA (Se activa si hay datos) ---
if archivo_subido is not None:
    # Si el banco sube su archivo real, lo leemos
    df = pd.read_csv(archivo_subido)
    st.success("Datos operativos cargados correctamente en el sistema.")
else:
    # MODO DEMO
    st.info("Modo de simulación activado: Generando 12.000 expedientes para demostración del modelo...")
    np.random.seed(0)
    df = pd.DataFrame({
        'ID_Cliente': range(1, 12001),
        'Importe_Deuda': np.round(np.random.uniform(50, 15000, 12000), 2),
        'Dias_Impago': np.random.randint(1, 90, 12000),
        'Prioridad_Banco': np.random.choice([1, 2, 3], 12000, p=[0.6, 0.3, 0.1]),
        'Llamadas_Previas': np.random.randint(0, 6, 12000)
    })
    # Lógica de mora para entrenar en modo demo
    prob_mora = (df['Dias_Impago'] / 90) * 0.6 + (df['Importe_Deuda'] / 15000) * 0.2 + (df['Prioridad_Banco'] / 3) * 0.2
    df['Llego_a_Mora'] = (prob_mora > np.random.uniform(0.3, 0.9, 12000)).astype(int)

# --- ENTRENAMIENTO Y PREDICCIÓN ---
X = df[['Importe_Deuda', 'Dias_Impago', 'Prioridad_Banco', 'Llamadas_Previas']]
if 'Llego_a_Mora' in df.columns: 
    Y = df['Llego_a_Mora']
    modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
    modelo.fit(X, Y)
    
    # Predecir riesgo
    df['Score_Urgencia'] = np.round(modelo.predict_proba(X)[:, 1] * 100, 2)
    df_final = df.sort_values(by='Score_Urgencia', ascending=False)

    # --- DASHBOARD VISUAL ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Expedientes a Gestionar", f"{len(df):,}", "Volumen diario")
    casos_criticos = len(df_final[df_final['Score_Urgencia'] > 80])
    col2.metric("Casos Nivel Crítico (>80% Riesgo)", f"{casos_criticos}", "Prioridad Operativa", delta_color="inverse")
    dinero_riesgo = df_final.head(casos_criticos)['Importe_Deuda'].sum()
    col3.metric("Capital en Máximo Riesgo", f"{dinero_riesgo:,.0f} €", "Asegurado por el sistema")

    st.subheader("Bandeja de Trabajo Priorizada")
    if st.button("Asignar casos urgentes a centralita"):
        st.success("Casos derivados a los gestores exitosamente.")
        
    st.dataframe(df_final[['ID_Cliente', 'Score_Urgencia', 'Dias_Impago', 'Importe_Deuda', 'Prioridad_Banco']], use_container_width=True)
