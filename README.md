# 💊 Droguerías VidaPlus - Agente Inteligente de Consulta Documental

## Descripción General

Este proyecto implementa un agente inteligente basado en Inteligencia Artificial capaz de responder preguntas en lenguaje natural utilizando como fuente de conocimiento la documentación interna de **Droguerías VidaPlus**, una cadena ficticia de droguerías ubicada en la ciudad de Armenia, Quindío.

El agente combina un paso de **triaje** (para decidir qué tipo de solicitud es) con **Retrieval-Augmented Generation (RAG)** sobre la base documental de la empresa, usando Claude (Anthropic) como modelo de lenguaje. Mantiene memoria de la conversación, por lo que entiende preguntas de seguimiento sin que el usuario tenga que repetir el contexto, y deriva a contacto humano cuando la solicitud lo requiere (excepciones, reclamos formales) o cuando no tiene información suficiente para responder.

> 🤖 **Desarrollo asistido por inteligencia artificial:** este proyecto se construyó de la mano de IA, utilizando **[Claude Code](https://claude.com/claude-code)** (Anthropic) como asistente de desarrollo para el diseño, la implementación, la depuración y la documentación. Todas las decisiones de negocio y el código final fueron revisados y validados por el autor.

### Base de Conocimiento

El agente consulta los siguientes documentos (en `docs/`):

- Manual Corporativo de Droguerías VidaPlus
- Manual de Promociones y Descuentos
- Programa Cliente VidaPlus
- Política de Domicilios
- Política de Cambios, Reembolsos y Garantías
- Política de Tratamiento de Datos Personales
- Preguntas Frecuentes (FAQ)
- Glosario Corporativo

### Interfaz

La interacción con el agente se realiza a través de una interfaz de chat web construida con Streamlit, que incluye:

- **Preguntas sugeridas** al iniciar, para orientar al usuario sobre lo que puede consultar.
- **Respuestas con efecto de escritura progresiva** (streaming).
- **Canales de contacto** y un **modo desarrollador** (para inspeccionar el triaje y cargar documentos) en el menú lateral.

![Interfaz del asistente Droguerías VidaPlus](src/img/interfaz.png)

Desde el **modo desarrollador** es posible ampliar la base de conocimiento cargando nuevos PDFs en caliente, sin reiniciar la aplicación. En el ejemplo, tras subir una guía de productos para mascotas, el agente pasa de no tener información sobre el tema a responder con base en el documento recién agregado:

![Carga de documentos en caliente desde el modo desarrollador](src/img/CargaArchivos.png)

---

## Cómo Funciona el Grafo

El agente está implementado como un grafo de estados (LangGraph) con tres decisiones de negocio encadenadas:

1. **¿Hay que entender esto en el contexto de la conversación?** (`contextualizar`) — si es la primera pregunta o un saludo, pasa igual; si es un seguimiento ("¿y en Laureles?"), la reescribe como una pregunta autónoma usando el historial. Además resuelve las referencias a días relativos ("hoy", "mañana") al día concreto de la semana, para que preguntas como "¿qué promoción hay hoy?" recuperen la información correcta.
2. **¿Qué tipo de solicitud es?** (`triaje`) — clasifica el mensaje en `AUTO_RESOLVER` (pregunta sobre políticas), `PEDIR_INFO` (mensaje ambiguo o tema fuera del alcance de la base documental) o `DERIVAR_CONTACTO` (excepción/reclamo que requiere un humano).
3. **¿De verdad se pudo resolver?** (`auto_resolver`) — busca en la base documental y califica si los fragmentos recuperados son suficientes antes de generar la respuesta; si no lo son, reenvía a `pedir_info` o, si detecta palabras de excepción/autorización, a `derivar_contacto`.

Todos los caminos terminan en `finalizar`, que guarda el turno en el historial de la conversación y arma el resumen de documentos consultados que se le muestra al usuario.

```mermaid
flowchart TD
    START([Inicio]) --> CTX[contextualizar]
    CTX --> TRI[triaje]
    TRI -->|AUTO_RESOLVER| RAG[auto_resolver]
    TRI -->|PEDIR_INFO| INFO[pedir_info]
    TRI -->|DERIVAR_CONTACTO| CONTACTO[derivar_contacto]
    RAG -->|info suficiente| FIN[finalizar]
    RAG -->|info insuficiente| INFO
    RAG -->|info insuficiente + palabras de excepción| CONTACTO
    INFO --> FIN
    CONTACTO --> FIN
    FIN --> END([Fin])
```

---

## Arquitectura de la Solución

- **Triaje**: clasificación de intención con salida estructurada (Claude + `pydantic`), con reglas explícitas y ejemplos por categoría.
- **RAG con calificación de relevancia**: los documentos se dividen en fragmentos, se indexan con embeddings locales (FAISS) y se recuperan por similitud; antes de generar la respuesta, un paso adicional evalúa si los fragmentos recuperados alcanzan para responder (en vez de inferirlo del texto de la respuesta generada).
- **Memoria conversacional**: el grafo persiste el estado entre preguntas de una misma sesión (checkpointer en memoria), y un nodo de contextualización reformula preguntas de seguimiento antes de clasificarlas.
- **Derivación a contacto humano**: para excepciones, autorizaciones especiales o reclamos formales, el agente muestra directamente los canales de contacto reales de la droguería.
- **Separación respuesta / auditoría**: el usuario solo ve la respuesta final y un resumen de los documentos consultados; el contenido crudo de cada fragmento recuperado se registra en logs, no se muestra en la conversación.

---

## Tecnologías y Herramientas Utilizadas

- **Python 3.12**
- **Streamlit** — interfaz de chat web (con soporte para cargar documentos nuevos en caliente y un "modo desarrollador" para inspeccionar el triaje)
- **LangChain** / `langchain-anthropic` / `langchain-community` / `langchain-text-splitters` / `langchain-classic`
- **LangGraph** (grafo de estados + memoria conversacional con `MemorySaver`)
- **Claude (Anthropic)** como modelo de lenguaje (`ChatAnthropic`)
- **FastEmbed** — embeddings locales y gratuitos, con un modelo multilingüe (`paraphrase-multilingual-MiniLM-L12-v2`) para trabajar en español
- **FAISS** — base de datos vectorial local
- **PyMuPDF** — lectura de los PDFs de la base de conocimiento
- **Pydantic** — validación y salidas estructuradas (triaje, calificación de relevancia)
- **python-dotenv** — configuración vía variables de entorno (local); `st.secrets` en el despliegue

---

## Instrucciones para Ejecutar el Proyecto

### Ejecución local

1. **Clonar el repositorio** y entrar a la carpeta del proyecto:
   ```bash
   git clone https://github.com/armandoagudelo/Challenge_Alura.git
   cd Challenge_Alura
   ```
2. **Crear y activar un entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate      # En Windows: venv\Scripts\activate
   ```
3. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configurar la clave de API** creando un archivo `.env` en la raíz del proyecto:
   ```env
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   Opcionalmente puedes ajustar `MODELO_CLAUDE` o `CARPETA_DOCUMENTOS` en el mismo archivo.
5. **Ejecutar la interfaz de chat** (Streamlit):
   ```bash
   streamlit run src/app.py
   ```
   También existe un modo de consola: `python src/rag.py`.

---

## Despliegue en Streamlit Community Cloud

La aplicación está desplegada en vivo aquí: **https://vidaplus.streamlit.app/**

El despliegue en [Streamlit Community Cloud](https://share.streamlit.io) (plan gratuito) es directo:

1. **Crear una cuenta gratuita** en https://share.streamlit.io.
2. **Autorizar el acceso a GitHub** para que Streamlit pueda leer tus repositorios.
3. **Seleccionar el repositorio público** y la rama a desplegar (`main`).
4. **Indicar el archivo principal**: `src/app.py`.
5. **Configurar los Secrets** (paso imprescindible): en *Advanced settings → Secrets*
   agregar la clave de API, ya que Streamlit Cloud **no** usa el archivo `.env`:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   La app puentea automáticamente `st.secrets` hacia `os.environ` al arrancar
   (ver `src/app.py`), por lo que `rag.py` sigue leyendo la clave con `os.getenv`.
6. **Darle un nombre / URL** y pulsar **Deploy**.

Notas adicionales:
- Streamlit Cloud instala las dependencias desde `requirements.txt`, así que este
  archivo debe estar completo y actualizado.
- Cada `push` a la rama desplegada (`main`) **redespliega la app automáticamente**.
- Si necesitas una versión específica de Python, puedes fijarla con un archivo
  `runtime.txt` (por ejemplo `python-3.12`) en la raíz del repositorio.
- El disco de Streamlit Cloud es efímero: la base vectorial FAISS se reconstruye en
  cada arranque a partir de los PDFs de `docs/` (no requiere persistencia externa).

---

## Ejemplos de Preguntas

El agente puede responder preguntas como las siguientes:

1. ¿Qué promociones hay los jueves?
2. ¿Qué beneficios ofrece el Plan Vida Premium?
3. ¿Los descuentos de los lunes aplican pagando con cualquier tarjeta?
4. ¿Puedo devolver un medicamento si ya no lo necesito?
5. ¿Qué sucede si mi pedido llega incompleto?
6. ¿Cuánto cuesta un domicilio?
7. ¿En qué zonas de Armenia realizan domicilios?
8. ¿Puedo solicitar medicamentos con fórmula médica a domicilio?
9. ¿Cómo puedo afiliarme al Programa Cliente VidaPlus?
10. ¿Qué información personal recopila Droguerías VidaPlus?
11. ¿Cómo puedo dejar de recibir promociones por correo o teléfono?
12. ¿Qué ocurre si un producto llega dañado durante el transporte?
13. ¿Los puntos del programa tienen fecha de vencimiento?
14. ¿Qué productos tienen descuento los viernes?
15. ¿Cuál es la diferencia entre un medicamento genérico y uno de marca?
16. Quiero una autorización especial para devolver un medicamento fuera de los plazos. *(ejemplo de derivación a contacto humano)*

---

## Ejemplos de Respuestas Generadas por el Agente

**Pregunta:** ¿Cuánto cuesta un domicilio?

> El costo del domicilio depende de la distancia entre la sede que despacha el pedido y el lugar de entrega:
>
> - Hasta 3 km: $4.000
> - Entre 3 y 6 km: $6.000
> - Más de 6 km: Desde $8.000
>
> El valor definitivo será informado al cliente antes de confirmar el pedido.
>
> **Documentos consultados:** Política de Domicilios.pdf, Preguntas Frecuentes (FAQ).pdf, Programa Cliente VidaPlus.pdf

**Pregunta de seguimiento (misma conversación):** ¿y en Laureles?

> El costo del domicilio depende de la distancia, no específicamente del sector. Laureles es un sector cubierto por la operación en Armenia; para el costo exacto se necesita la distancia desde la sede hasta la dirección del cliente en esa zona.

**Pregunta:** Quiero una autorización especial para devolver un medicamento fuera de los plazos.

> Esta solicitud (urgencia MEDIANA) requiere la atención de un agente humano. Por favor contáctanos directamente:
> 📞 Teléfono / WhatsApp: +57 315 900 00 00
> 📧 Correo: droguerias.vidaplus@gmail.com
> Con gusto te atenderemos personalmente.

**Pregunta:** Necesito ayuda.

> Lo siento, no tengo información sobre ese tema. Con gusto puedo ayudarte con: domicilios, promociones y descuentos, el Programa Cliente VidaPlus, cambios, reembolsos y garantías, o el tratamiento de tus datos personales. ¿Sobre cuál te gustaría preguntar?
