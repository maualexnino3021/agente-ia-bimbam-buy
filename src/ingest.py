"""
Ingesta de la base de conocimiento:
1. Carga los documentos .txt de data/base_conocimiento/
2. Los divide en fragmentos (chunking)
3. Genera embeddings locales y los indexa en FAISS
4. Persiste el índice en vectorstore/

Ejecutar con:
    python -m src.ingest
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.llm import get_embeddings

# Nombres legibles para cada archivo fuente (usados al citar la fuente)
NOMBRES_DOCUMENTOS = {
    "01_programa_afiliados.txt": "Programa de Afiliados",
    "02_politica_reembolsos_devoluciones.txt": "Política de Reembolsos y Devoluciones",
    "03_faq_metodos_pago.txt": "Preguntas Frecuentes sobre Métodos de Pago",
    "04_manual_garantia_productos.txt": "Manual de Garantía de Productos",
    "05_guia_tiempos_costos_envio.txt": "Guía de Tiempos y Costos de Envío",
}


def cargar_documentos(directorio: Path):
    """Carga todos los .txt del directorio de base de conocimiento."""
    documentos = []
    archivos = sorted(directorio.glob("*.txt"))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt en '{directorio}'. "
            "Verifica que la base de conocimiento esté en su lugar."
        )

    for archivo in archivos:
        loader = TextLoader(str(archivo), encoding="utf-8")
        docs = loader.load()
        nombre_legible = NOMBRES_DOCUMENTOS.get(archivo.name, archivo.stem)
        for doc in docs:
            doc.metadata["source_name"] = nombre_legible
            doc.metadata["source_file"] = archivo.name
        documentos.extend(docs)

    return documentos


def dividir_documentos(documentos):
    """Divide los documentos en fragmentos manejables para el retriever."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


def construir_indice() -> FAISS:
    """Pipeline completo: cargar -> dividir -> embeddings -> FAISS -> guardar."""
    print(f"📂 Cargando documentos desde: {settings.knowledge_base_dir}")
    documentos = cargar_documentos(settings.knowledge_base_dir)
    print(f"   {len(documentos)} documento(s) cargado(s).")

    print("✂️  Dividiendo documentos en fragmentos...")
    fragmentos = dividir_documentos(documentos)
    print(f"   {len(fragmentos)} fragmento(s) generado(s).")

    print("🧠 Generando embeddings (esto puede tardar la primera vez)...")
    embeddings = get_embeddings()

    print("🗂️  Construyendo índice FAISS...")
    indice = FAISS.from_documents(fragmentos, embeddings)

    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    indice.save_local(str(settings.vectorstore_dir))
    print(f"✅ Índice guardado en: {settings.vectorstore_dir}")

    return indice


if __name__ == "__main__":
    construir_indice()
