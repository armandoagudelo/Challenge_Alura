# -*- coding: utf-8 -*-
"""
Agente de RAG + Triaje con Claude (Anthropic)

Primera versión = Consolda, basado en el curso RAG y agentes de AI de Alura Latam (2024).

Adaptado desde un notebook de Colab que usaba Gemini. Cambios principales:
- El LLM ahora es Claude (ChatAnthropic) en vez de Gemini.
- Anthropic no ofrece API de embeddings, así que los embeddings para el
  RAG se generan localmente con un modelo FastEmbedEmbeddings (model_name="sentence-transformers/all-MiniLM-L6-v2")
- Los PDFs ya no se leen de una carpeta local "./docs" (configurable).
- Consola para poder chatear con el agente.
"""

import logging
import operator
import os
import re
from pathlib import Path
from typing import Annotated, Literal, List, Dict, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

logger = logging.getLogger("vidaplus_rag")


# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent  # carpeta que contiene src/ y docs/

load_dotenv(RAIZ_PROYECTO / ".env")  # lee el .env de la raíz del proyecto

# Log del detalle de citaciones (ver mostrar_respuesta): va a un archivo, no a
# la consola, para que la consola muestre solo la respuesta al usuario.
CARPETA_LOGS = RAIZ_PROYECTO / "logs"
CARPETA_LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    filename=CARPETA_LOGS / "rag.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "No se encontró ANTHROPIC_API_KEY. Crea un archivo .env con "
        "ANTHROPIC_API_KEY=sk-ant-... o expórtala como variable de entorno."
    )

CARPETA_DOCUMENTOS = Path(os.getenv("CARPETA_DOCUMENTOS", RAIZ_PROYECTO / "docs"))
MODELO_CLAUDE = os.getenv("MODELO_CLAUDE", "claude-haiku-4-5-20251001")

# Canales de contacto humano para los casos que el agente no puede resolver.
TELEFONO_CONTACTO = "+57 315 900 00 00"
CORREO_CONTACTO = "droguerias.vidaplus@gmail.com"

# Nombres de PDFs cargados en caliente por el usuario (ver
# agregar_documentos_pdf). El triaje los suma a sus temas conocidos para no
# clasificarlos como ambiguos.
DOCUMENTOS_AGREGADOS: List[str] = []

# thread_id fijo para el loop de consola (memoria entre preguntas). Un
# frontend real debería generar uno único por sesión de usuario.
SESSION_ID = "sesion-consola"

# Modelo de LLM (Claude).
llm = ChatAnthropic(
    model=MODELO_CLAUDE,
    max_retries=2,
    api_key=ANTHROPIC_API_KEY,
)

# ---------------------------------------------------------------------------
# Triaje: clasifica el mensaje del usuario
# ---------------------------------------------------------------------------

PROMPT_TRIAJE = """
Eres un especialista en triaje del Service Desk de atención al cliente de Droguerías
VidaPlus (una cadena de droguerías en Armenia, Quindío).
Dado el mensaje del usuario, devuelve SÓLO un JSON con:
{
    "decision": "AUTO_RESOLVER" | "PEDIR_INFO" | "DERIVAR_CONTACTO",
    "urgencia": "BAJA" | "MEDIANA" | "ALTA",
    "campos_faltantes": ["..."]
}

La base de conocimiento de la empresa cubre estos documentos y temas:
- Manual Corporativo: sedes, horarios, medios de pago, convenios, servicios generales.
- Política de Domicilios: cobertura, horarios, tiempos y costos de entrega, medicamentos
  con fórmula médica a domicilio, entregas fallidas, productos faltantes.
- Manual de Promociones y Descuentos: calendario semanal de descuentos, condiciones de
  acumulación, exclusiones.
- Programa Cliente VidaPlus: planes de afiliación (Esencial, Plus, Premium), puntos,
  beneficios, domicilio gratuito.
- Política de Cambios, Reembolsos y Garantías: devoluciones, productos que no admiten
  cambio, garantías por defecto de fabricación, tiempos de reembolso.
- Política de Tratamiento de Datos Personales: privacidad, derechos del titular,
  comunicaciones comerciales, cómo dejar de recibir promociones.

Reglas:
- AUTO_RESOLVER: Cualquier pregunta identificable dentro de alguno de los temas anteriores,
  aunque esté formulada de manera general o coloquial. No necesitas saber de antemano si el
  documento contiene la respuesta exacta: basta con que el tema sea reconocible para que el
  sistema de búsqueda (RAG) lo intente resolver.
  Ejemplos reales de clientes que SÍ son AUTO_RESOLVER:
  - "¿Cuánto cuesta un domicilio?" (tema: Política de Domicilios)
  - "¿En qué zonas de Armenia realizan domicilios?" (tema: Política de Domicilios)
  - "¿Puedo solicitar medicamentos con fórmula médica a domicilio?" (tema: Política de
    Domicilios)
  - "¿Cómo puedo dejar de recibir promociones por correo o teléfono?" (tema: Tratamiento de
    Datos Personales / comunicaciones comerciales)
  - "¿Puedo devolver un medicamento que ya abrí?" (tema: Cambios, Reembolsos y Garantías)
  - "¿Qué beneficios tiene el Plan Vida Premium?" (tema: Programa Cliente VidaPlus)
  - "¿Qué promoción hay los jueves?" (tema: Promociones y Descuentos)
- PEDIR_INFO: Únicamente cuando el mensaje sea tan ambiguo que no se pueda relacionar con
  NINGUNO de los temas anteriores, ni siquiera de forma general.
  Ejemplos:
  - "Necesito ayuda."
  - "Tengo un problema con mi pedido." (no dice cuál problema ni con qué tema se relaciona)
  - "¿Me pueden colaborar?"
- DERIVAR_CONTACTO: Solicitudes de excepciones, autorización especial, quejas formales que
  requieren intervención humana, o cuando el usuario pide explícitamente hablar con una
  persona o radicar un reclamo.
  Ejemplos:
  - "Quiero una autorización especial para devolver un medicamento fuera de los plazos."
  - "Quiero radicar una queja formal por un domiciliario."

Ante la duda entre AUTO_RESOLVER y PEDIR_INFO, prefiere AUTO_RESOLVER: es mejor dejar que
el sistema busque en las políticas y responda "No lo sé" si no encuentra nada, que pedirle
información adicional a un cliente cuya pregunta ya es identificable.
Analiza el mensaje y decide la acción más adecuada.
"""

class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "DERIVAR_CONTACTO"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


chain_triaje = llm.with_structured_output(TriajeOut)


def _prompt_triaje() -> str:
    if not DOCUMENTOS_AGREGADOS:
        return PROMPT_TRIAJE
    extra = "\n".join(f"- {n}" for n in DOCUMENTOS_AGREGADOS)
    return (
        PROMPT_TRIAJE
        + "\nDocumentos adicionales cargados por el usuario (trátalos como temas válidos "
        f"para AUTO_RESOLVER):\n{extra}\n"
    )


def triaje(mensaje: str) -> Dict:
    salida: TriajeOut = chain_triaje.invoke(
        [
            SystemMessage(content=_prompt_triaje()),
            HumanMessage(content=mensaje),
        ]
    )
    return salida.model_dump()


# ---------------------------------------------------------------------------
# RAG: carga de documentos, embeddings, vectorstore y búsqueda
# ---------------------------------------------------------------------------

def limpiar_texto(texto: str) -> str:
    """Quita ruido de la extracción de PDF (espacios de ancho cero, viñetas,
    saltos de línea repetidos) que degrada la calidad de los embeddings."""
    texto = texto.replace("​", " ")
    texto = re.sub(r"●\s*", "- ", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{2,}", "\n\n", texto)
    return texto.strip()


def cargar_documentos(carpeta: Path):
    docs = []
    if not carpeta.exists():
        carpeta.mkdir(parents=True, exist_ok=True)
        print(f"Se creó la carpeta '{carpeta}'. Coloca tus PDFs ahí y vuelve a correr el script.")
        return docs

    for archivo in carpeta.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(archivo))
            documentos_archivo = loader.load()
            for doc in documentos_archivo:
                doc.page_content = limpiar_texto(doc.page_content)
            docs.extend(documentos_archivo)
            print(f"Archivo cargado: {archivo.name}")
        except Exception as e:
            print(f"Error cargando archivo {archivo.name}: {e}")

    print(f"Total de documentos cargados: {len(docs)}")
    return docs


docs = cargar_documentos(CARPETA_DOCUMENTOS)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs) if docs else []

# Embeddings locales, gratuitos. Modelo multilingüe (no el MiniLM en inglés):
# los documentos y preguntas están en español, y un modelo solo-inglés da
# embeddings de baja calidad para ese idioma.
modelo_embeddings = FastEmbedEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# "similarity" (top-k) en vez de "similarity_score_threshold": con MiniLM
# los puntajes no están bien calibrados entre preguntas distintas.
RETRIEVER_KWARGS = {"search_type": "similarity", "search_kwargs": {"k": 6}}

vectorstore = None
retriever = None
if chunks:
    vectorstore = FAISS.from_documents(chunks, modelo_embeddings)
    retriever = vectorstore.as_retriever(**RETRIEVER_KWARGS)
else:
    print("No hay documentos para indexar todavía; el RAG responderá 'No lo sé' hasta que agregues PDFs.")


def agregar_documentos_pdf(rutas) -> int:
    """Indexa PDFs adicionales en caliente sobre el mismo vectorstore. Devuelve
    cuántos fragmentos se agregaron."""
    global vectorstore, retriever
    nuevos_docs = []
    nombres = []
    for ruta in rutas:
        docs_archivo = PyMuPDFLoader(str(ruta)).load()
        for doc in docs_archivo:
            doc.page_content = limpiar_texto(doc.page_content)
        if docs_archivo:
            nuevos_docs.extend(docs_archivo)
            nombres.append(Path(ruta).name)

    nuevos_chunks = splitter.split_documents(nuevos_docs) if nuevos_docs else []
    if not nuevos_chunks:
        return 0

    if vectorstore is None:
        vectorstore = FAISS.from_documents(nuevos_chunks, modelo_embeddings)
        retriever = vectorstore.as_retriever(**RETRIEVER_KWARGS)
    else:
        vectorstore.add_documents(nuevos_chunks)
    DOCUMENTOS_AGREGADOS.extend(nombres)
    return len(nuevos_chunks)

prompt_rag = ChatPromptTemplate(
    [
        (
            "system",
            "Eres el especialista en RR.HH. de la empresa. "
            "Responde siempre utilizando los conocimientos del contexto entregado. "
            "Si no hay información sobre la pregunta en el contexto, responde solo 'No lo sé'.",
        ),
        ("human", "Contexto: {context}\nPregunta del empleado: {input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt_rag)


class RelevanciaOut(BaseModel):
    documentos_suficientes: bool


PROMPT_RELEVANCIA = """
Eres un evaluador de calidad para un sistema de RAG. Dada una pregunta de un cliente y
los fragmentos de documentos recuperados, decide si esos fragmentos contienen información
suficiente para responder la pregunta de forma concreta.
Responde "documentos_suficientes": true solo si de verdad se puede construir una respuesta
útil con ese contenido; si los fragmentos son irrelevantes o solo mencionan el tema de
pasada sin responderlo, responde false.
"""

chain_relevancia = llm.with_structured_output(RelevanciaOut)


def calificar_relevancia(pregunta: str, documentos: list) -> bool:
    """Reemplaza el chequeo frágil de comparar la respuesta generada contra el
    string literal "No lo sé"."""
    contexto = "\n\n".join(doc.page_content for doc in documentos)
    salida: RelevanciaOut = chain_relevancia.invoke(
        [
            SystemMessage(content=PROMPT_RELEVANCIA),
            HumanMessage(content=f"Pregunta: {pregunta}\n\nFragmentos recuperados:\n{contexto}"),
        ]
    )
    return salida.documentos_suficientes


def busqueda_de_respuestas_RAG(pregunta: str) -> Dict:
    if retriever is None:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}

    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}

    if not calificar_relevancia(pregunta, documentos_relacionados):
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}

    answer = document_chain.invoke({"input": pregunta, "context": documentos_relacionados})

    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True,
    }


# ---------------------------------------------------------------------------
# Agente con LangGraph:
# contextualizar -> triaje -> auto_resolver / pedir_info / derivar_contacto -> finalizar
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    pregunta: str
    pregunta_efectiva: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    documentos_encontrados: Optional[bool]
    documentos_consultados: List[str]
    rag_exito: bool
    accion_final: str
    info_por_rag_insuficiente: bool  # ver nodo_pedir_info
    historial: Annotated[List[Dict[str, str]], operator.add]  # ver nodo_finalizar


PROMPT_CONTEXTUALIZAR = """
Eres un asistente que reformula preguntas de seguimiento para que se puedan entender por sí
solas, sin necesitar el historial de la conversación.
Dado el historial de turnos anteriores y el mensaje nuevo del usuario, devuelve una única
pregunta autónoma que capture la intención real del usuario, incorporando el contexto
necesario del historial (por ejemplo, el tema o la zona mencionados antes) SOLO cuando el
mensaje nuevo realmente depende de ese contexto para entenderse.
Si el mensaje nuevo ya es autónomo y no depende del historial, devuélvelo tal cual, sin
agregar ni quitar información.
Si el mensaje nuevo es un saludo, una despedida, un agradecimiento, o cualquier mensaje
neutro que no tenga relación con el historial (ej. "hola", "gracias", "listo"), devuélvelo
tal cual. NUNCA lo reemplaces por una pregunta anterior del historial: un saludo no es una
pregunta de seguimiento.
Responde ÚNICAMENTE con el mensaje resultante, sin explicaciones adicionales.
"""


def nodo_contextualizar(state: AgentState) -> AgentState:
    """Reformula la pregunta usando el historial y reinicia info_por_rag_insuficiente
    (si no se resetea, un campo no reescrito en el turno actual conserva el valor
    persistido del turno anterior)."""
    historial = state.get("historial") or []
    if not historial:
        return {"pregunta_efectiva": state["pregunta"], "info_por_rag_insuficiente": False}

    texto_historial = "\n".join(
        f"Usuario: {turno['pregunta']}\nAsistente: {turno['respuesta']}" for turno in historial
    )
    respuesta = llm.invoke(
        [
            SystemMessage(content=PROMPT_CONTEXTUALIZAR),
            HumanMessage(
                content=f"Historial:\n{texto_historial}\n\nPregunta nueva: {state['pregunta']}"
            ),
        ]
    )
    return {
        "pregunta_efectiva": respuesta.content.strip(),
        "info_por_rag_insuficiente": False,
    }


def nodo_triaje(state: AgentState) -> AgentState:
    return {"triaje": triaje(state["pregunta_efectiva"])}


def nodo_auto_resolver(state: AgentState) -> AgentState:
    respuesta_RAG = busqueda_de_respuestas_RAG(state["pregunta_efectiva"])

    update: AgentState = {
        "respuesta": respuesta_RAG["respuesta"],
        "citaciones": respuesta_RAG["citaciones"],
        "rag_exito": respuesta_RAG["documentos_encontrados"],
    }

    update["accion_final"] = "AUTO_RESOLVER" if respuesta_RAG["documentos_encontrados"] else "PEDIR_INFO"
    if not respuesta_RAG["documentos_encontrados"]:
        update["info_por_rag_insuficiente"] = True
    return update


def nodo_pedir_info(state: AgentState) -> AgentState:
    """Mensaje distinto según el origen: RAG sin resultados vs. triaje ambiguo."""
    if state.get("info_por_rag_insuficiente"):
        mensaje = "No encontré información específica sobre eso en nuestras políticas."
    else:
        mensaje = (
            "Cuéntame en qué puedo ayudarte: domicilios, promociones y descuentos, el "
            "Programa Cliente VidaPlus, cambios/reembolsos/garantías, o el tratamiento de "
            "tus datos personales."
        )

    return {
        "respuesta": mensaje,
        "citaciones": [],
        "accion_final": "PEDIR_INFO",
    }


def nodo_derivar_contacto(state: AgentState) -> AgentState:
    """Deriva a un agente humano (no crea ningún ticket real)."""
    tri = state["triaje"]
    mensaje = (
        f"Esta solicitud (urgencia {tri['urgencia']}) requiere la atención de un agente "
        "humano. Por favor contáctanos directamente:\n"
        f"📞 Teléfono / WhatsApp: {TELEFONO_CONTACTO}\n"
        f"📧 Correo: {CORREO_CONTACTO}\n"
        "Con gusto te atenderemos personalmente."
    )
    return {
        "respuesta": mensaje,
        "citaciones": [],
        "accion_final": "DERIVAR_CONTACTO",
    }


def nodo_finalizar(state: AgentState) -> AgentState:
    """Punto único antes de END: guarda el turno en el historial y arma el
    resumen de documentos consultados."""
    citaciones = state.get("citaciones") or []
    documentos = sorted({
        Path(c.metadata.get("file_path", "desconocido")).name for c in citaciones
    })
    return {
        "documentos_consultados": documentos,
        "historial": [{"pregunta": state["pregunta"], "respuesta": state["respuesta"]}],
    }


def arista_decision_triaje(state: AgentState) -> str:
    tri = state["triaje"]
    if tri["decision"] == "AUTO_RESOLVER":
        return "rag"
    elif tri["decision"] == "PEDIR_INFO":
        return "info"
    else:
        return "contacto"


KEYWORDS_DERIVAR_CONTACTO = [
    "aprobación", "aprobar", "excepción", "liberación", "autorización",
    "autorizar", "abrir ticket", "acceso especial",
]


def arista_decision_rag(state: AgentState) -> str:
    if state["rag_exito"]:
        return "ok"
    if any(keyword in state["pregunta_efectiva"].lower() for keyword in KEYWORDS_DERIVAR_CONTACTO):
        return "contacto"
    return "info"


def construir_grafo():
    workflow = StateGraph(AgentState)

    workflow.add_node("contextualizar", nodo_contextualizar)
    workflow.add_node("triaje", nodo_triaje)
    workflow.add_node("auto_resolver", nodo_auto_resolver)
    workflow.add_node("pedir_info", nodo_pedir_info)
    workflow.add_node("derivar_contacto", nodo_derivar_contacto)
    workflow.add_node("finalizar", nodo_finalizar)

    workflow.add_edge(START, "contextualizar")
    workflow.add_edge("contextualizar", "triaje")

    workflow.add_conditional_edges(
        "triaje",
        arista_decision_triaje,
        {"rag": "auto_resolver", "info": "pedir_info", "contacto": "derivar_contacto"},
    )

    workflow.add_conditional_edges(
        "auto_resolver",
        arista_decision_rag,
        {"info": "pedir_info", "contacto": "derivar_contacto", "ok": "finalizar"},
    )

    workflow.add_edge("pedir_info", "finalizar")
    workflow.add_edge("derivar_contacto", "finalizar")
    workflow.add_edge("finalizar", END)

    return workflow.compile(checkpointer=MemorySaver())


grafo = construir_grafo()


def mostrar_respuesta(resultado: Dict):
    """Consola de cara al usuario: solo respuesta + documentos consultados,
    en secciones separadas. El detalle de triaje y el contenido crudo de las
    citaciones se registran en logs/rag.log, no se imprimen aquí."""
    tri = resultado["triaje"]
    logger.info(
        "Triaje | decision=%s urgencia=%s faltantes=%s",
        tri["decision"], tri["urgencia"], tri["campos_faltantes"],
    )
    for i, citacion in enumerate(resultado.get("citaciones") or []):
        logger.info(
            "Citación %d | Documento: %s | Contenido: %s",
            i + 1,
            citacion.metadata.get("file_path", "desconocido"),
            citacion.page_content.replace("\n", " "),
        )

    separador = "-" * 60
    print(separador)
    print("RESPUESTA")
    print(separador)
    print(resultado["respuesta"])

    documentos_consultados = resultado.get("documentos_consultados") or []
    if documentos_consultados:
        print()
        print(separador)
        print("DOCUMENTOS CONSULTADOS")
        print(separador)
        for nombre in documentos_consultados:
            print(f"- {nombre}")
    print(separador)


# ---------------------------------------------------------------------------
# Loop de consola
# ---------------------------------------------------------------------------

def main():
    print("=== Agente RAG con Claude ===")
    print(f"Carpeta de documentos: {CARPETA_DOCUMENTOS.resolve()}")
    print("Escribe tu pregunta (o 'salir' para terminar).\n")

    config = {"configurable": {"thread_id": SESSION_ID}}

    while True:
        pregunta = input("Tú: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        resultado = grafo.invoke({"pregunta": pregunta}, config=config)
        mostrar_respuesta(resultado)
        print()


if __name__ == "__main__":
    main()