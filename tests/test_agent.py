"""
Pruebas básicas del pipeline RAG.

Nota: estas pruebas ejercitan la ingesta y el retriever (no requieren API
key de LLM). Para probar `AgenteBimBamBuy.responder()` de punta a punta sí
necesitas configurar OPENAI_API_KEY o GOOGLE_API_KEY en tu .env.

Ejecutar con:
    pytest -q
"""
from src.config import settings
from src.ingest import cargar_documentos, dividir_documentos
from src.vectorstore import obtener_retriever


def test_cargar_documentos_encuentra_los_5_archivos():
    documentos = cargar_documentos(settings.knowledge_base_dir)
    fuentes = {doc.metadata["source_file"] for doc in documentos}
    assert len(fuentes) == 5


def test_chunking_genera_fragmentos():
    documentos = cargar_documentos(settings.knowledge_base_dir)
    fragmentos = dividir_documentos(documentos)
    assert len(fragmentos) > len(documentos)


def test_retriever_devuelve_resultados_relevantes():
    retriever = obtener_retriever(k=3)
    resultados = retriever.invoke("¿Cuánto tiempo tengo para devolver un producto?")
    assert len(resultados) > 0
    fuentes = {doc.metadata.get("source_name") for doc in resultados}
    assert any("Reembolsos" in f for f in fuentes if f)
