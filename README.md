# 💊 Droguerías VidaPlus - Agente Inteligente de Consulta Documental

## Descripción General

Este proyecto implementa un agente inteligente basado en Inteligencia Artificial capaz de responder preguntas en lenguaje natural utilizando como fuente de conocimiento la documentación interna de **Droguerías VidaPlus**, una cadena ficticia de droguerías ubicada en la ciudad de Armenia, Quindío.

El objetivo del proyecto es demostrar cómo un sistema basado en **Retrieval-Augmented Generation (RAG)** puede ayudar a colaboradores y clientes a consultar información corporativa sin necesidad de revisar manualmente múltiples documentos.

En lugar de buscar información en políticas, manuales o programas de fidelización, el usuario simplemente realiza una pregunta y el agente recupera la información más relevante para generar una respuesta precisa y contextualizada.

---

## Objetivos del Proyecto

- Construir una base de conocimiento empresarial estructurada.
- Implementar un agente capaz de comprender preguntas en lenguaje natural.
- Recuperar información desde múltiples documentos utilizando RAG.
- Generar respuestas claras, consistentes y fundamentadas en la documentación.
- Simular un caso de uso empresarial real para organizaciones que manejan grandes volúmenes de información.

---

## Base de Conocimiento

El agente consulta los siguientes documentos:

- Manual Corporativo de Droguerías VidaPlus
- Manual de Promociones y Descuentos
- Programa Cliente VidaPlus
- Política de Domicilios
- Política de Cambios, Reembolsos y Garantías
- Política de Tratamiento de Datos Personales
- Preguntas Frecuentes (FAQ)
- Glosario Corporativo

---

## Tecnologías Utilizadas

- Python
- LangChain
- OpenAI API
- ChromaDB (Base de datos vectorial)
- Embeddings
- Retrieval-Augmented Generation (RAG)

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

---

## Objetivo del Agente

El propósito del agente es reducir el tiempo que los usuarios invierten buscando información en múltiples documentos, proporcionando respuestas rápidas, precisas y fundamentadas en la documentación oficial de la empresa.

---

## Caso de Uso

Este proyecto simula un escenario empresarial en el que una organización almacena gran cantidad de información en documentos PDF.

En lugar de buscar manualmente dentro de dichos documentos, los usuarios pueden realizar preguntas en lenguaje natural y obtener respuestas respaldadas por la base documental mediante técnicas de Recuperación Aumentada por Generación (RAG).

Este tipo de solución puede aplicarse a empresas de sectores como salud, banca, retail, manufactura, educación y cualquier organización que gestione documentación corporativa de gran volumen.