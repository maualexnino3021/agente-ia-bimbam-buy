"""
Fábrica de modelos de lenguaje (LLM).

Permite alternar entre OpenAI y Google Gemini sin tocar el resto del
código, únicamente cambiando la variable de entorno LLM_PROVIDER.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings


def get_llm(temperature: float = 0.3) -> BaseChatModel:
    """Devuelve una instancia de chat model según LLM_PROVIDER."""
    settings.validate()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    raise ValueError(f"Proveedor de LLM no soportado: {settings.llm_provider}")


def get_embeddings():
    """Devuelve el modelo de embeddings según EMBEDDINGS_PROVIDER.

    - "openai" (por defecto): usa la API de OpenAI (text-embedding-3-small).
      Es la opción recomendada para desplegar en Render: no descarga ningún
      modelo pesado durante el build ni consume memoria adicional en tiempo
      de ejecución, lo cual es clave en el plan free (512 MB de RAM).
    - "local": usa un modelo de sentence-transformers ejecutado en tu propia
      máquina, sin costo por token, pero requiere instalar las dependencias
      de requirements-local-embeddings.txt (incluye PyTorch, varios cientos
      de MB) y usa bastante más memoria y tiempo de carga.
    """
    settings.validate()

    if settings.embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embeddings_model)
