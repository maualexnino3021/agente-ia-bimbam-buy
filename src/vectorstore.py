"""
Utilidades para cargar el índice FAISS ya construido, generándolo
automáticamente si todavía no existe.
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS

from src.config import settings
from src.llm import get_embeddings


def cargar_vectorstore() -> FAISS:
    """Carga el índice FAISS desde disco. Si no existe, lo construye."""
    index_file = settings.vectorstore_dir / "index.faiss"

    if not index_file.exists():
        print("⚠️  No se encontró un índice existente. Generando uno nuevo...")
        from src.ingest import construir_indice

        return construir_indice()

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(settings.vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def obtener_retriever(k: int | None = None):
    """Devuelve un retriever listo para usar en la cadena RAG."""
    vectorstore = cargar_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k or settings.top_k})
