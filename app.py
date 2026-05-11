import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier

# Configuración profesional
st.set_page_config(page_title="OPPLUS - Triaje Inteligente", layout="wide")
st.title("Sistema de Triaje GYAR - Equipo Houston")

# --- SISTEMA DE USUARIOS SIMULADO ---
USUARIOS = {
    "director": {"pwd": "1234", "rol": "Director"},
    "gestor1": {"pwd": "1234", "rol": "Gestor 1", "rango": (0, 100)},
    "gestor2": {"pwd": "1234", "rol": "Gestor 2", "rango": (100, 200)},
    "gestor3": {"pwd": "1234", "rol": "Gestor 3", "rango": (200, 300)},
    "gestor4": {"pwd": "1234", "rol": "Gestor 4", "rango": (300, 400)}
}

# --- PERSISTENCIA DE DATOS EN LA SESIÓN ---
if 'df_operativo' not in st.session_state:
    st.session_state.df_operativo = None
if 'usuario_actual' not in st.session_state:
    st.session_state.usuario_actual = None
if 'rol_actual' not in st.session_state:
    st.session_state.rol_actual = None
    
# --- 2. MOTOR DE IA (RE-TRIADORA) ---
def ejecutar_ia_triaje(df):
    """Entrena y predice el riesgo basado en el estado actual de los datos"""
    columnas_ia = ['Importe_Deuda', 'Dias_Impago', 'Prioridad_Banco', 'Llamadas_Previas']
    
    # Simulación de entrenamiento (solo si no viene con etiquetas de mora)
    df_entreno = df.copy()
    
    if 'Llego_a_Mora' not in df_entreno.columns:
        # 1. Calculamos el riesgo base (Días e Importe)
        base_riesgo = (df_entreno['Dias_Impago'] / 90) * 0.6 + \
                      (df_entreno['Importe_Deuda'] / df_entreno['Importe_Deuda'].max()) * 0.4
        
        # 2. Aplicamos la regla de saturación (3 o más llamadas = hachazo)
        df_entreno['Saturado'] = df_entreno['Llamadas_Previas'] >= 3
        
        # 3. Si está saturado, restamos 0.7. Si no, restamos solo 0.05 por llamada.
        prob = np.where(df_entreno['Saturado'], 
                        base_riesgo - 0.7, 
                        base_riesgo - (df_entreno['Llamadas_Previas'] * 0.05))
        
        # 4. Marcamos como "Mora" para que el Árbol de Decisión aprenda el patrón
        df_entreno['Llego_a_Mora'] = (prob > 0.5).astype(int)
    
    # El modelo se entrena FUERA del if para que funcione siempre
    modelo = DecisionTreeClassifier(max_depth=4)
    modelo.fit(df_entreno[columnas_ia], df_entreno['Llego_a_Mora'])
    
    # Generamos el Score base de la IA
    prob_ia = modelo.predict_proba(df[columnas_ia])[:, 1] * 100
    
    # AJUSTE: Restamos un 1% por cada llamada para que el Score sea dinámico y se mueva siempre
    df['Score_Urgencia'] = np.round(prob_ia - (df['Llamadas_Previas'] * 1.0), 2)
    
    # Ordenamos por Score (de mayor a menor) y por Llamadas (de menor a mayor para desempatar)
    return df.sort_values(by=['Score_Urgencia', 'Llamadas_Previas'], ascending=[False, True])
    
# --- BARRA LATERAL CON LOGIN Y EXPORTACIÓN ---
with st.sidebar:
    if st.session_state.usuario_actual is None:
        st.header("🔐 Acceso al Sistema")
        user_input = st.text_input("Usuario")
        pass_input = st.text_input("Contraseña", type="password")
        
        if st.button("Iniciar Sesión"):
            if user_input in USUARIOS and USUARIOS[user_input]["pwd"] == pass_input:
                st.session_state.usuario_actual = user_input
                st.session_state.rol_actual = USUARIOS[user_input]["rol"]
                st.success("Acceso concedido")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Credenciales incorrectas")
    else:
        st.header(f"👤 {st.session_state.rol_actual}")
        if st.button("Cerrar Sesión"):
            st.session_state.usuario_actual = None
            st.session_state.rol_actual = None
            st.rerun()
            
        # La descarga es un privilegio SOLO del Director
        if st.session_state.rol_actual == "Director":
            st.divider()
            st.header("💾 Gestión de Progreso")
            if st.session_state.df_operativo is not None:
                csv_data = st.session_state.df_operativo.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Extraer Informe de Gestión", data=csv_data, file_name="base_actualizada.csv", mime="text/csv")

# --- BLOQUEO SI NO HAY USUARIO ---
if st.session_state.usuario_actual is None:
    st.info("👈 Por favor, inicie sesión en el menú lateral para acceder a su panel.")
    st.stop()


# --- 1. CARGA DE EXPEDIENTES (SOLO DIRECTOR) ---
if st.session_state.df_operativo is None:
    if st.session_state.rol_actual == "Director":
        st.markdown("### 1. Inicialización de la Jornada")
        archivo_subido = st.file_uploader("Suba el CSV con la base de datos completa", type=["csv"])

        if archivo_subido is not None:
            df_inicial = pd.read_csv(archivo_subido)
            if 'Gestionado_Hoy' not in df_inicial.columns:
                df_inicial['Gestionado_Hoy'] = False  
            if 'Llamadas_Previas' not in df_inicial.columns:
                df_inicial['Llamadas_Previas'] = 0
            st.session_state.df_operativo = df_inicial
            st.rerun()
    else:
        st.warning("⏳ Esperando a que el Director de Operaciones cargue la base de datos del día...")
    # st.stop() aquí es vital, para que no intente ejecutar lo de abajo si no hay datos.
    st.stop() 


# --- 3. PANEL DE CONTROL Y GESTIÓN ---
if st.session_state.df_operativo is not None:
    # Calculamos la IA para todos
    df_maestro = ejecutar_ia_triaje(st.session_state.df_operativo)
    df_solo_pendientes = df_maestro[df_maestro['Gestionado_Hoy'] == False]
    total_datos = len(st.session_state.df_operativo)
    pendientes_ahora = len(df_solo_pendientes)

    # ==========================================
    # ==========================================
    # VISTA 1: EL DIRECTOR
    # ==========================================
    if st.session_state.rol_actual == "Director":
        st.subheader("Cuadro de Mando Integral")
        
        # 1. MEJORA VISUAL: Barra de progreso de la campaña
        gestionados_hoy = total_datos - pendientes_ahora
        porcentaje_avance = int((gestionados_hoy / total_datos) * 100) if total_datos > 0 else 0
        st.progress(porcentaje_avance, text=f"Avance Global de la Jornada: {porcentaje_avance}%")
        
        st.divider()
        
        # Métricas principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volumen Inicial", f"{total_datos:,}")
        
        delta_p = f"-{gestionados_hoy} tramitados" if gestionados_hoy > 0 else None
        c2.metric("Pendiente de Gestión", f"{pendientes_ahora:,}", delta=delta_p)
        
        capital_en_vuelo = df_solo_pendientes['Importe_Deuda'].sum()
        c3.metric("Riesgo Vivo Asignado", f"{capital_en_vuelo:,.0f} €")

        capital_liberado = st.session_state.df_operativo[st.session_state.df_operativo['Gestionado_Hoy'] == True]['Importe_Deuda'].sum()
        delta_exito = "Impacto Operativo" if capital_liberado > 0 else None
        c4.metric("Capital Liberado", f"{capital_liberado:,.0f} €", delta=delta_exito, delta_color="normal")
        
        st.divider()

        # --- NUEVA SECCIÓN: KPIs OBJETIVO DEL RETO ---
        st.markdown("#### Cumplimiento de KPIs Estratégicos (Objetivo OPPLUS)")
        kpi1, kpi2 = st.columns(2)
        
        # Cálculo KPI 1: % Gestionados antes de 60 días
        df_gestionados = st.session_state.df_operativo[st.session_state.df_operativo['Gestionado_Hoy'] == True]
        if len(df_gestionados) > 0:
            gestionados_antes_60 = len(df_gestionados[df_gestionados['Dias_Impago'] <= 60])
            pct_antes_60 = (gestionados_antes_60 / len(df_gestionados)) * 100
        else:
            pct_antes_60 = 0.0
            
        kpi1.metric("Visibilidad ANS: % Gestionados <= 60 días", f"{pct_antes_60:.1f} %", delta="Monitorización en tiempo real")
        
        # Cálculo KPI 2: Productividad
        num_gestores = 4 # Plantilla base de simulación
        productividad = len(df_gestionados) / num_gestores
        kpi2.metric("Productividad: Expedientes / Gestor / Día", f"{productividad:.1f}", delta="Media de equipo activo")

        st.divider()

        # 2. MEJORA VISUAL: Columnas con Gráfico y Tabla
        col_grafico, col_tabla = st.columns([1, 1])
        
        with col_grafico:
            st.markdown("#### Concentración de Riesgo (Top 50 Críticos)")
            st.write("Visualización de la deuda de los casos con mayor Score de Urgencia.")
            df_grafico = df_solo_pendientes.head(50).set_index('ID_Cliente')['Importe_Deuda']
            st.bar_chart(df_grafico, color="#004481")
            
        with col_tabla:
            st.markdown("#### Cola de Enrutamiento Activa")
            st.write("Los gestores están recibiendo estos expedientes en tiempo real.")
            st.dataframe(df_solo_pendientes[['ID_Cliente', 'Score_Urgencia', 'Importe_Deuda', 'Dias_Impago']].head(400), use_container_width=True)

    # ==========================================
    # VISTA 2: LOS GESTORES (PUESTO DE TRABAJO)
    # ==========================================
    else:
        rango_inicio, rango_fin = USUARIOS[st.session_state.usuario_actual]["rango"]
        st.subheader(f"Bandeja Operativa: Tramo Asignado ({rango_inicio} - {rango_fin})")
        
        # 3. MEJORA VISUAL: Meta diaria del gestor (Gamificación)
        st.info("Objetivo de Productividad Diario: Mantener la cola de su tramo despejada para asegurar el ANS de 60 días.")
        
        # Filtramos la tabla para que este gestor solo vea su tramo
        df_vista_mia = df_solo_pendientes.iloc[rango_inicio:rango_fin].copy()
        
        if len(df_vista_mia) == 0:
            st.success("Tramo operativo completado. Excelente trabajo. A la espera de nueva reasignación por parte del sistema.")
        else:
            df_vista_mia['Gestión Cerrada'] = False
            df_vista_mia['Contacto Fallido'] = False

            df_editado = st.data_editor(
                df_vista_mia[['Gestión Cerrada', 'Contacto Fallido', 'ID_Cliente', 'Score_Urgencia', 'Llamadas_Previas', 'Importe_Deuda', 'Dias_Impago']],
                use_container_width=True,
                hide_index=True,
                disabled=['ID_Cliente', 'Score_Urgencia', 'Llamadas_Previas', 'Importe_Deuda', 'Dias_Impago']
            )

            st.markdown("<br>", unsafe_allow_html=True) # Espaciado limpio

            if st.button("Sincronizar Operaciones"):
                with st.spinner("Sincronizando registros con el servidor central..."):
                    finalizados = df_editado[df_editado['Gestión Cerrada'] == True]['ID_Cliente'].values
                    reintentos = df_editado[df_editado['Contacto Fallido'] == True]['ID_Cliente'].values

                    for idx in st.session_state.df_operativo.index:
                        cid = st.session_state.df_operativo.at[idx, 'ID_Cliente']
                        if cid in finalizados:
                            st.session_state.df_operativo.at[idx, 'Gestionado_Hoy'] = True
                        if cid in reintentos:
                            st.session_state.df_operativo.at[idx, 'Llamadas_Previas'] += 1

                    st.toast("Operación registrada. Actualizando bandeja de trabajo.")
                    time.sleep(1.5)
                st.rerun()
