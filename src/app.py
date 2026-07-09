# -*- coding: utf-8 -*-
"""
Interfaz de chat en Streamlit para el agente RAG de Droguerías VidaPlus.

Ejecutar con: streamlit run src/app.py
"""

import base64
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="VidaPlus - Asistente", page_icon="💊", layout="centered")

# Streamlit Cloud no usa .env; las claves se configuran en el panel "Secrets"
# de la app (st.secrets). Puenteamos hacia os.environ antes de importar
# rag.py, que sigue leyendo la clave con os.getenv vía python-dotenv.
if "ANTHROPIC_API_KEY" not in os.environ:
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

try:
    with st.spinner("Cargando la base de conocimiento..."):
        import rag
except RuntimeError as e:
    st.error(str(e))
    st.stop()


RUTA_BANNER = Path(__file__).parent / "img" / "banner_web.jpg"
EMOJIS_URGENCIA = {"BAJA": "🟢", "MEDIANA": "🟡", "ALTA": "🔴"}

ESTILOS = """
<style>
.hero {
    position: relative;
    width: 100%;
    min-height: 190px;
    border-radius: 16px;
    overflow: hidden;
    margin: 0.25rem 0 1.5rem 0;
    background-size: cover;
    background-position: center 35%;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.15);
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        90deg,
        rgba(16, 52, 40, 0.88) 0%,
        rgba(16, 52, 40, 0.60) 45%,
        rgba(16, 52, 40, 0.10) 100%
    );
}
.hero-content {
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 190px;
    padding: 1.6rem 1.8rem;
}
.hero-eyebrow {
    color: #b9e6cd;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.hero-title {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
}
.hero-sub {
    color: #e9f5ee;
    font-size: 0.98rem;
    margin-top: 0.5rem;
    max-width: 34rem;
    text-shadow: 0 1px 6px rgba(0, 0, 0, 0.4);
}
[data-testid="stChatMessage"] {
    border-radius: 14px;
}
</style>
"""


@st.cache_data
def _imagen_base64(ruta: str) -> str:
    return base64.b64encode(Path(ruta).read_bytes()).decode()


def render_banner():
    st.markdown(ESTILOS, unsafe_allow_html=True)
    try:
        fondo = f"background-image:url('data:image/jpeg;base64,{_imagen_base64(str(RUTA_BANNER))}')"
    except FileNotFoundError:
        fondo = "background-color:#12362a"
    st.markdown(
        f"""
        <div class="hero" style="{fondo}">
          <div class="hero-content">
            <div class="hero-eyebrow">💊 Farmacia · Armenia, Quindío</div>
            <div class="hero-title">Droguerías VidaPlus</div>
            <div class="hero-sub">
              Asistente virtual — pregúntame sobre domicilios, promociones, el
              Programa Cliente VidaPlus, cambios y garantías, o el tratamiento
              de tus datos personales.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_triaje_badge(triaje):
    if not triaje:
        return
    emoji = EMOJIS_URGENCIA.get(triaje.get("urgencia"), "")
    st.caption(f"{emoji} Decisión: {triaje.get('decision')} · Urgencia: {triaje.get('urgencia')}")


def render_documentos(documentos_consultados):
    if not documentos_consultados:
        return
    with st.expander(f"Fuentes consultadas ({len(documentos_consultados)})"):
        for nombre in documentos_consultados:
            st.markdown(f"- {nombre}")


def render_accion_final(accion_final, respuesta):
    if accion_final == "PEDIR_INFO":
        st.info(respuesta)
    elif accion_final == "DERIVAR_CONTACTO":
        st.warning(respuesta)
    else:
        st.markdown(respuesta)


render_banner()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    modo_desarrollador = st.toggle(
        "Modo desarrollador",
        value=False,
        help="Muestra el triaje de cada respuesta y permite cargar documentos nuevos.",
    )

    if modo_desarrollador:
        st.divider()
        st.subheader("Agregar documentos")
        subidos = st.file_uploader(
            "Sube PDFs para ampliar la base de conocimiento",
            type="pdf",
            accept_multiple_files=True,
        )
        if subidos:
            procesados = st.session_state.setdefault("archivos_procesados", set())
            nuevos = [f for f in subidos if f.name not in procesados]
            if nuevos:
                rutas = []
                for f in nuevos:
                    ruta = Path(tempfile.gettempdir()) / f.name
                    ruta.write_bytes(f.getbuffer())
                    rutas.append(ruta)
                with st.spinner("Procesando documentos..."):
                    fragmentos = rag.agregar_documentos_pdf(rutas)
                for f in nuevos:
                    procesados.add(f.name)
                    st.session_state.setdefault("archivos_agregados", []).append(f.name)
                st.success(f"Se agregaron {len(nuevos)} documento(s) ({fragmentos} fragmentos).")

    st.divider()
    st.subheader("Documentos disponibles")
    for archivo in sorted(rag.CARPETA_DOCUMENTOS.glob("*.pdf")):
        st.caption(archivo.name)
    for nombre in st.session_state.get("archivos_agregados", []):
        st.caption(f"{nombre} (agregado)")

    st.divider()
    if st.button("Reiniciar conversación"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        if mensaje["role"] == "assistant":
            render_accion_final(mensaje["accion_final"], mensaje["respuesta"])
            if modo_desarrollador:
                render_triaje_badge(mensaje.get("triaje"))
            render_documentos(mensaje.get("documentos_consultados"))
        else:
            st.markdown(mensaje["content"])

pregunta = st.chat_input("Escribe tu pregunta...")
if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            resultado = rag.grafo.invoke({"pregunta": pregunta}, config=config)

        render_accion_final(resultado["accion_final"], resultado["respuesta"])
        if modo_desarrollador:
            render_triaje_badge(resultado.get("triaje"))
        render_documentos(resultado.get("documentos_consultados"))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "respuesta": resultado["respuesta"],
            "documentos_consultados": resultado.get("documentos_consultados"),
            "triaje": resultado.get("triaje"),
            "accion_final": resultado["accion_final"],
        }
    )
