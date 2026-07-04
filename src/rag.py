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

import os
from pathlib import Path
from typing import Literal, List, Dict, Optional, TypedDict

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
from langgraph.graph import START, END, StateGraph


# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent  # carpeta que contiene src/ y docs/

load_dotenv(RAIZ_PROYECTO / ".env")  # lee el .env de la raíz del proyecto

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "No se encontró ANTHROPIC_API_KEY. Crea un archivo .env con "
        "ANTHROPIC_API_KEY=sk-ant-... o expórtala como variable de entorno."
    )

CARPETA_DOCUMENTOS = Path(os.getenv("CARPETA_DOCUMENTOS", RAIZ_PROYECTO / "docs"))
MODELO_CLAUDE = os.getenv("MODELO_CLAUDE", "claude-haiku-4-5-20251001")

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
Eres un especialista en triaje del Service Desk para políticas internas.
Dado el mensaje del usuario, devuelve SÓLO un JSON con:
{
    "decision": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_TICKET",
    "urgencia": "BAJA" | "MEDIANA" | "ALTA",
    "campos_faltantes": ["..."]
}
Reglas:
- AUTO_RESOLVER: Preguntas claras sobre las reglas o procedimientos descritos en las políticas
  (Ej.: "¿Puedo reembolsar el internet para mi oficina en casa?").
- PEDIR_INFO: Mensajes imprecisos o sin información para identificar el tema o el contexto
  (Ej.: "Necesito ayuda con una política").
- ABRIR_TICKET: Solicitudes de excepciones, autorización, aprobación o acceso especial, o
  cuando el usuario solicita explícitamente abrir un ticket
  (Ej.: "Quiero una excepción para trabajar remotamente durante 5 días").
Analiza el mensaje y decide la acción más adecuada.
"""

class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


chain_triaje = llm.with_structured_output(TriajeOut)


def triaje(mensaje: str) -> Dict:
    salida: TriajeOut = chain_triaje.invoke(
        [
            SystemMessage(content=PROMPT_TRIAJE),
            HumanMessage(content=mensaje),
        ]
    )
    return salida.model_dump()


# ---------------------------------------------------------------------------
# RAG: carga de documentos, embeddings, vectorstore y búsqueda
# ---------------------------------------------------------------------------

def cargar_documentos(carpeta: Path):
    docs = []
    if not carpeta.exists():
        carpeta.mkdir(parents=True, exist_ok=True)
        print(f"Se creó la carpeta '{carpeta}'. Coloca tus PDFs ahí y vuelve a correr el script.")
        return docs

    for archivo in carpeta.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(archivo))
            docs.extend(loader.load())
            print(f"Archivo cargado: {archivo.name}")
        except Exception as e:
            print(f"Error cargando archivo {archivo.name}: {e}")

    print(f"Total de documentos cargados: {len(docs)}")
    return docs


docs = cargar_documentos(CARPETA_DOCUMENTOS)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs) if docs else []

# Embeddings locales (no requieren API key). El modelo se descarga una sola
# vez la primera vez que se corre el script.
modelo_embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

retriever = None
if chunks:
    vectorstore = FAISS.from_documents(chunks, modelo_embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4},
    )
else:
    print("No hay documentos para indexar todavía; el RAG responderá 'No lo sé' hasta que agregues PDFs.")

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


def busqueda_de_respuestas_RAG(pregunta: str) -> Dict:
    if retriever is None:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}

    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}

    answer = document_chain.invoke({"input": pregunta, "context": documentos_relacionados})

    if answer.rstrip(".!?") == "No lo sé":
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}

    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True,
    }


# ---------------------------------------------------------------------------
# Agente con LangGraph: triaje -> auto_resolver / pedir_info / abrir_ticket
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    documentos_encontrados: Optional[bool]
    rag_exito: bool
    accion_final: str


def nodo_triaje(state: AgentState) -> AgentState:
    return {"triaje": triaje(state["pregunta"])}


def nodo_auto_resolver(state: AgentState) -> AgentState:
    respuesta_RAG = busqueda_de_respuestas_RAG(state["pregunta"])

    update: AgentState = {
        "respuesta": respuesta_RAG["respuesta"],
        "citaciones": respuesta_RAG["citaciones"],
        "rag_exito": respuesta_RAG["documentos_encontrados"],
    }

    update["accion_final"] = "AUTO_RESOLVER" if respuesta_RAG["documentos_encontrados"] else "pedir_info"
    return update


def nodo_pedir_info(state: AgentState) -> AgentState:
    return {
        "respuesta": "Necesito más información sobre tu pedido.",
        "citaciones": [],
        "accion_final": "PEDIR_INFO",
    }


def nodo_abrir_ticket(state: AgentState) -> AgentState:
    tri = state["triaje"]
    return {
        "respuesta": f"Abrir ticket con urgencia {tri['urgencia']}. Pedido: {state['pregunta']}.",
        "citaciones": [],
        "accion_final": "ABRIR_TICKET",
    }


def arista_decision_triaje(state: AgentState) -> str:
    tri = state["triaje"]
    if tri["decision"] == "AUTO_RESOLVER":
        return "rag"
    elif tri["decision"] == "PEDIR_INFO":
        return "info"
    else:
        return "ticket"


KEYWORDS_ABRIR_TICKET = [
    "aprobación", "aprobar", "excepción", "liberación", "autorización",
    "autorizar", "abrir ticket", "acceso especial",
]


def arista_decision_rag(state: AgentState) -> str:
    if state["rag_exito"]:
        return "ok"
    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
        return "ticket"
    return "info"


def construir_grafo():
    workflow = StateGraph(AgentState)

    workflow.add_node("triaje", nodo_triaje)
    workflow.add_node("auto_resolver", nodo_auto_resolver)
    workflow.add_node("pedir_info", nodo_pedir_info)
    workflow.add_node("abrir_ticket", nodo_abrir_ticket)

    workflow.add_edge(START, "triaje")

    workflow.add_conditional_edges(
        "triaje",
        arista_decision_triaje,
        {"rag": "auto_resolver", "info": "pedir_info", "ticket": "abrir_ticket"},
    )

    workflow.add_conditional_edges(
        "auto_resolver",
        arista_decision_rag,
        {"info": "pedir_info", "ticket": "abrir_ticket", "ok": END},
    )

    workflow.add_edge("pedir_info", END)
    workflow.add_edge("abrir_ticket", END)

    return workflow.compile()


grafo = construir_grafo()


def mostrar_respuesta(resultado: Dict):
    tri = resultado["triaje"]
    print(f"DECISION: {tri['decision']} | URGENCIA: {tri['urgencia']} | FALTANTES: {tri['campos_faltantes']}")
    print(f"RESPUESTA: {resultado['respuesta']}")

    citaciones = resultado.get("citaciones") or []
    for i, citacion in enumerate(citaciones):
        contenido = citacion.page_content.replace("\n", " ")
        print(f"  - CITACION {i + 1}:")
        print(f"    Documento: {citacion.metadata.get('file_path', 'desconocido')}")
        print(f"    Contenido: {contenido}")


# ---------------------------------------------------------------------------
# Loop de consola
# ---------------------------------------------------------------------------

def main():
    print("=== Agente RAG con Claude ===")
    print(f"Carpeta de documentos: {CARPETA_DOCUMENTOS.resolve()}")
    print("Escribe tu pregunta (o 'salir' para terminar).\n")

    while True:
        pregunta = input("Tú: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        resultado = grafo.invoke({"pregunta": pregunta})
        mostrar_respuesta(resultado)
        print()


if __name__ == "__main__":
    main()