"""
Interfaz de chat web (Streamlit) para el Agente de Soporte de BimBam Buy.

Ejecutar con:
    streamlit run app_streamlit.py
"""
import streamlit as st

from src.agent import AgenteBimBamBuy

st.set_page_config(
    page_title="Agente de Soporte — BimBam Buy",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agente de Soporte — BimBam Buy")
st.caption(
    "Pregúntame sobre devoluciones, garantías, métodos de pago, envíos o el "
    "programa de afiliados. Respondo solo con base en nuestros documentos "
    "oficiales."
)


@st.cache_resource(show_spinner="Cargando agente y base de conocimiento...")
def cargar_agente() -> AgenteBimBamBuy:
    return AgenteBimBamBuy()


agente = cargar_agente()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {
            "rol": "assistant",
            "contenido": "¡Hola! 👋 Soy el agente de soporte de BimBam Buy. "
            "¿En qué puedo ayudarte hoy?",
            "fuentes": [],
        }
    ]

with st.sidebar:
    st.header("⚙️ Opciones")
    if st.button("🔄 Reiniciar conversación"):
        agente.reiniciar_conversacion()
        st.session_state.mensajes = [
            {
                "rol": "assistant",
                "contenido": "Conversación reiniciada. ¿En qué puedo ayudarte?",
                "fuentes": [],
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("**Base de conocimiento activa:**")
    st.markdown(
        "- Programa de Afiliados\n"
        "- Política de Reembolsos y Devoluciones\n"
        "- FAQ de Métodos de Pago\n"
        "- Manual de Garantía de Productos\n"
        "- Guía de Tiempos y Costos de Envío"
    )

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if mensaje.get("fuentes"):
            st.caption("📎 Fuentes: " + ", ".join(mensaje["fuentes"]))

pregunta = st.chat_input("Escribe tu pregunta aquí...")

if pregunta:
    st.session_state.mensajes.append(
        {"rol": "user", "contenido": pregunta, "fuentes": []}
    )
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimiento..."):
            respuesta = agente.responder(pregunta)
        st.markdown(respuesta.contenido)
        if respuesta.fuentes:
            st.caption("📎 Fuentes: " + ", ".join(respuesta.fuentes))

    st.session_state.mensajes.append(
        {
            "rol": "assistant",
            "contenido": respuesta.contenido,
            "fuentes": respuesta.fuentes,
        }
    )
