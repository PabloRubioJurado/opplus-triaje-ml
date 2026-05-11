import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier

# Configuración profesional
st.set_page_config(page_title="OPPLUS - Triaje Operativo", layout="wide")
st.title("Sistema de Enrutamiento Inteligente - Equipo Houston")

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
        st.header("Acceso al Sistema")
        user_input = st.text_input("Usuario")
        pass_input = st.text_input("Contraseña", type="password")
        
        if st.button("Iniciar Sesión"):
            if user_input in USUARIOS and USUARIOS[user_input]["pwd"] == pass_input:
                st.session_state.usuario_actual = user_input
                st.session_state.rol_actual = USUARIOS[user_input]["rol"]
                st.success("Acceso concedido.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Credenciales incorrectas.")
    else:
        st.header(f"Usuario: {st.session_state.rol_actual}")
        if st.button("Cerrar Sesión"):
            st.session_state.usuario_actual = None
            st.session_state.rol_actual = None
            st.rerun()
            
        # La descarga es un privilegio SOLO del Director
        if st.session_state.rol_actual == "Director":
            st.divider()
            st.header("Exportación de Datos")
            if st.session_state.df_operativo is not None:
                csv_data = st.session_state.df_operativo.to_csv(index=False).encode('utf-8')
                st.download_button(label="Extraer Informe de Gestión", data=csv_data, file_name="base_actualizada.csv", mime="text/csv")

# --- BLOQUEO SI NO HAY USUARIO ---
if st.session_state.usuario_actual is None:
    st.info("Por favor, inicie sesión en el menú lateral para acceder a su panel operativo.")
    st.stop()


# --- 1. CARGA DE EXPEDIENTES (SOLO DIRECTOR) ---
if st.session_state.df_operativo is None:
    if st.session_state.rol_actual == "Director":
        st.markdown("### Inicialización de la Jornada")
        archivo_subido = st.file_uploader("Suba el archivo CSV con la cartera base", type=["csv"])

        if archivo_subido is not None:
            df_inicial = pd.read_csv(archivo_subido)
            if 'Gestionado_Hoy' not in df_inicial.columns:
                df_inicial['Gestionado_Hoy'] = False  
            if 'Llamadas_Previas' not in df_inicial.columns:
                df_inicial['Llamadas_Previas'] = 0
            st.session_state.df_operativo = df_inicial
            st.rerun()
    else:
        st.warning("El sistema está a la espera de la asignación de cartera por parte de Dirección.")
        
        # Botón de escape para el gestor atrapado
        if st.button("Cerrar Sesión y Volver al Inicio"):
            st.session_state.usuario_actual = None
            st.session_state.rol_actual = None
            st.rerun()
            
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
    # VISTA 1: EL DIRECTOR
    # ==========================================
    if st.session_state.rol_actual == "Director":
        st.subheader("Cuadro de Mando Integral")
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volumen Inicial", f"{total_datos:,}")
        
        gestionados_hoy = total_datos - pendientes_ahora
        delta_p = f"-{gestionados_hoy} tramitados" if gestionados_hoy > 0 else None
        c2.metric("Pendiente de Gestión", f"{pendientes_ahora:,}", delta=delta_p)
        
        capital_en_vuelo = df_solo_pendientes['Importe_Deuda'].sum()
        c3.metric("Riesgo Vivo Asignado", f"{capital_en_vuelo:,.0f} €")

        capital_liberado = st.session_state.df_operativo[st.session_state.df_operativo['Gestionado_Hoy'] == True]['Importe_Deuda'].sum()
        delta_exito = "Impacto Operativo" if capital_liberado > 0 else None
        c4.metric("Capital Liberado", f"{capital_liberado:,.0f} €", delta=delta_exito, delta_color="normal")
        
        st.info("El equipo se encuentra trabajando en sus respectivos tramos. Utilice el menú lateral para exportar la auditoría en formato CSV.")

    # ==========================================
    # VISTA 2: LOS GESTORES
    # ==========================================
    else:
        rango_inicio, rango_fin = USUARIOS[st.session_state.usuario_actual]["rango"]
        st.subheader(f"Bandeja Operativa: Tramo Asignado ({rango_inicio} - {rango_fin})")
        
        # Filtramos la tabla para que este gestor solo vea su tramo
        df_vista_mia = df_solo_pendientes.iloc[rango_inicio:rango_fin].copy()
        
        if len(df_vista_mia) == 0:
            st.success("Tramo operativo completado. A la espera de nueva reasignación por parte del sistema.")
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
                    time.sleep(2)
                st.rerun()
