"""
Configuración centralizada del proyecto.

Lee las variables de entorno (desde .env) y expone valores por defecto
razonables para que el proyecto funcione incluso sin un .env completo
(salvo por las API keys del LLM, que sí son obligatorias).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Carga el archivo .env ubicado en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # Proveedor de LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Gemini
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Embeddings
    # Proveedor de embeddings: "openai" (por defecto, liviano, recomendado para
    # desplegar en Render) o "local" (gratis con sentence-transformers, pero
    # requiere instalar requirements-local-embeddings.txt y consume mucha más
    # memoria/tiempo de build; ver README).
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    embeddings_model: str = os.getenv(
        "EMBEDDINGS_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Parámetros RAG
    chunk_size: int = _get_int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)
    top_k: int = _get_int("TOP_K", 4)

    # Rutas
    knowledge_base_dir: Path = BASE_DIR / os.getenv(
        "KNOWLEDGE_BASE_DIR", "data/base_conocimiento"
    )
    vectorstore_dir: Path = BASE_DIR / os.getenv("VECTORSTORE_DIR", "vectorstore")

    def validate(self) -> None:
        """Lanza un error claro si falta configuración crítica."""
        if self.llm_provider not in {"openai", "gemini"}:
            raise ValueError(
                f"LLM_PROVIDER inválido: '{self.llm_provider}'. "
                "Usa 'openai' o 'gemini'."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "Falta OPENAI_API_KEY en tu .env (LLM_PROVIDER=openai)."
            )
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError(
                "Falta GOOGLE_API_KEY en tu .env (LLM_PROVIDER=gemini)."
            )
        if self.embeddings_provider not in {"openai", "local"}:
            raise ValueError(
                f"EMBEDDINGS_PROVIDER inválido: '{self.embeddings_provider}'. "
                "Usa 'openai' o 'local'."
            )
        # Los embeddings de OpenAI son necesarios siempre que EMBEDDINGS_PROVIDER
        # sea "openai", incluso si el LLM de chat usado es Gemini.
        if self.embeddings_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "Falta OPENAI_API_KEY en tu .env (EMBEDDINGS_PROVIDER=openai). "
                "Es obligatoria para generar los embeddings del RAG, "
                "independientemente del LLM_PROVIDER que uses."
            )


settings = Settings()
