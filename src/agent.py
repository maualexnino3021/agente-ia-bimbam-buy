"""
Agente conversacional RAG de BimBam Buy.

Expone la clase `AgenteBimBamBuy`, que encapsula:
- el retriever (búsqueda por similitud sobre FAISS)
- el LLM (OpenAI o Gemini, según configuración)
- el historial de conversación (memoria de corto plazo, en memoria del proceso)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT, format_context
from src.vectorstore import obtener_retriever


@dataclass
class RespuestaAgente:
    contenido: str
    fuentes: list[str] = field(default_factory=list)


class AgenteBimBamBuy:
    """Agente de soporte al cliente basado en RAG para BimBam Buy."""

    def __init__(self, top_k: int | None = None, temperature: float = 0.3):
        self.retriever = obtener_retriever(k=top_k)
        self.llm = get_llm(temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", RAG_PROMPT_TEMPLATE),
            ]
        )
        self.historial: list[HumanMessage | AIMessage] = []

    def _formatear_historial(self) -> str:
        if not self.historial:
            return "(sin mensajes previos)"
        lineas = []
        for msg in self.historial[-6:]:  # últimos 3 turnos (6 mensajes)
            rol = "Cliente" if isinstance(msg, HumanMessage) else "Agente"
            lineas.append(f"{rol}: {msg.content}")
        return "\n".join(lineas)

    def responder(self, pregunta: str) -> RespuestaAgente:
        """Responde una pregunta del usuario usando RAG y actualiza la memoria."""
        documentos = self.retriever.invoke(pregunta)
        contexto = format_context(documentos)
        historial_texto = self._formatear_historial()

        mensajes = self.prompt.format_messages(
            context=contexto,
            chat_history=historial_texto,
            question=pregunta,
        )

        respuesta = self.llm.invoke(mensajes)
        contenido = respuesta.content

        # Actualiza memoria conversacional
        self.historial.append(HumanMessage(content=pregunta))
        self.historial.append(AIMessage(content=contenido))

        fuentes = sorted(
            {doc.metadata.get("source_name", "desconocida") for doc in documentos}
        )
        return RespuestaAgente(contenido=contenido, fuentes=fuentes)

    def reiniciar_conversacion(self) -> None:
        """Limpia el historial de la sesión actual."""
        self.historial = []
