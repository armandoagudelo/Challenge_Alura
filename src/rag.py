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
import re
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
Eres un especialista en triaje del Service Desk de atención al cliente de Droguerías
VidaPlus (una cadena de droguerías en Armenia, Quindío).
Dado el mensaje del usuario, devuelve SÓLO un JSON con:
{
    "decision": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_TICKET",
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
- ABRIR_TICKET: Solicitudes de excepciones, autorización especial, quejas formales que
  requieren intervención humana, o cuando el usuario pide explícitamente abrir un ticket.
  Ejemplos:
  - "Quiero una autorización especial para devolver un medicamento fuera de los plazos."
  - "Quiero radicar una queja formal por un domiciliario."

Ante la duda entre AUTO_RESOLVER y PEDIR_INFO, prefiere AUTO_RESOLVER: es mejor dejar que
el sistema busque en las políticas y responda "No lo sé" si no encuentra nada, que pedirle
información adicional a un cliente cuya pregunta ya es identificable.
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

def limpiar_texto(texto: str) -> str:
    # PyMuPDF deja espacios de ancho cero ("​") pegados a las viñetas y
    # múltiples saltos de línea que fragmentan las oraciones. Un modelo de
    # embeddings local y pequeño como MiniLM es sensible a este ruido: lo
    # limpiamos antes de indexar para mejorar la calidad de la búsqueda.
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

# Embeddings locales (no requieren API key). El modelo se descarga una sola
# vez la primera vez que se corre el script.
# Se usa un modelo multilingüe (no el MiniLM en inglés) porque los documentos
# y las preguntas están en español: un modelo entrenado solo en inglés genera
# embeddings de baja calidad para texto en español, lo que hace que el
# retriever no encuentre los fragmentos correctos aunque el chunking sea
# bueno. Este modelo sigue siendo gratuito, local y liviano (~220 MB).
modelo_embeddings = FastEmbedEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

retriever = None
if chunks:
    vectorstore = FAISS.from_documents(chunks, modelo_embeddings)
    # "similarity" en vez de "similarity_score_threshold": con un modelo local
    # pequeño como MiniLM los puntajes de similitud no están bien calibrados
    # entre preguntas distintas, así que un umbral fijo descarta respuestas
    # válidas o deja pasar coincidencias débiles. Top-k puro es más confiable.
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
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