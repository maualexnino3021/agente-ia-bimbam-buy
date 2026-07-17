# 📘 GUIA_CHALLENGE_ALURA_AGENTE — Agente IA de Soporte BimBam Buy

## 1. Objetivo del challenge

Construir un **agente conversacional de IA** capaz de responder preguntas de
soporte al cliente de una tienda ficticia de e-commerce, **BimBam Buy**,
apoyándose **únicamente** en una base de conocimiento documental interna
(políticas, manuales y guías de la empresa), usando una arquitectura
**RAG (Retrieval-Augmented Generation)**.

El agente debe:

- Responder con información **fiel a los documentos** (evitar alucinaciones).
- Indicar de qué documento proviene la respuesta (trazabilidad / citas).
- Mantener un **tono de marca** consistente: cercano, claro, dinámico y con
  un toque divertido, sin perder profesionalismo (tal como define el propio
  documento de Identidad del Programa de Afiliados).
- Reconocer cuándo una pregunta **no puede responderse** con la base de
  conocimiento disponible, y decirlo explícitamente en lugar de inventar.
- Sostener una conversación con memoria de corto plazo (seguimiento de
  contexto en la misma sesión).

## 2. El caso de negocio: BimBam Buy

BimBam Buy es una tienda de e-commerce que opera en LATAM. Su equipo de
soporte necesita un agente que pueda resolver consultas frecuentes sin
escalar cada caso a una persona humana, cubriendo 5 áreas de conocimiento:

| # | Documento | Temas clave |
|---|---|---|
| 1 | **Programa de Afiliados** | elegibilidad, comisiones, atribución de ventas, pagos a afiliados, material permitido/no permitido, suspensión |
| 2 | **Política de Reembolsos y Devoluciones** | plazos, casos elegibles/no elegibles, flujo de atención, reembolsos parciales, tiempos de resolución |
| 3 | **FAQ de Métodos de Pago** | métodos disponibles, seguridad, prevención de fraude, facturación, pagos duplicados |
| 4 | **Manual de Garantía de Productos** | cobertura, exclusiones, procedimiento de evaluación, tiempos de respuesta |
| 5 | **Guía de Tiempos y Costos de Envío** | tipos de envío, tiempos estimados, costos, cobertura geográfica, incidencias logísticas |

Estos 5 documentos viven en `data/base_conocimiento/` y son la **única
fuente de verdad** que el agente puede citar.

## 3. Arquitectura de la solución

```
                ┌────────────────────────┐
                │  Base de conocimiento   │
                │  (5 documentos .txt)    │
                └───────────┬─────────────┘
                            │  1. Ingesta (src/ingest.py)
                            │     - Carga documentos
                            │     - Chunking (RecursiveCharacterTextSplitter)
                            │     - Embeddings (sentence-transformers)
                            ▼
                ┌────────────────────────┐
                │   Vector Store (FAISS)  │
                │      vectorstore/       │
                └───────────┬─────────────┘
                            │  2. Retrieval (src/agent.py)
                            │     - Búsqueda por similitud (top-k)
                            ▼
        ┌───────────────────────────────────────┐
        │     Prompt de sistema + contexto        │
        │     recuperado + historial de chat       │
        │             (src/prompts.py)             │
        └───────────────────┬───────────────────┘
                            │  3. Generación
                            ▼
                ┌────────────────────────┐
                │   LLM (OpenAI/Gemini)   │
                │      (src/llm.py)        │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  Respuesta al usuario   │
                │ (CLI o Streamlit chat)  │
                └────────────────────────┘
```

### Componentes

- **`src/ingest.py`**: lee los `.txt` de `data/base_conocimiento/`, los divide
  en fragmentos (`chunk_size=1000`, `chunk_overlap=150`) y genera un índice
  FAISS persistido en `vectorstore/`.
- **`src/vectorstore.py`**: helper para cargar/crear el índice FAISS de forma
  segura.
- **`src/llm.py`**: fábrica de modelos. Permite elegir el proveedor
  (`openai` o `gemini`) mediante la variable de entorno `LLM_PROVIDER`.
- **`src/prompts.py`**: contiene el *system prompt* con las reglas de tono y
  de anti-alucinación, y la plantilla de RAG.
- **`src/agent.py`**: arma la cadena conversacional (retriever + LLM +
  memoria) y expone una función `responder(pregunta, historial)`.
- **`main_cli.py`** / **`app_streamlit.py`**: dos interfaces de usuario sobre
  el mismo agente.

## 4. Reglas del agente (system prompt)

El prompt de sistema (ver `src/prompts.py`) obliga al modelo a:

1. Responder **solo** con base en el contexto recuperado de los documentos.
2. Si la información no está en el contexto, decir claramente que no cuenta
   con esa información y sugerir contactar a soporte humano.
3. Citar el documento de origen al final de la respuesta (ej. *"Fuente:
   Política de Reembolsos y Devoluciones"*).
4. Mantener el tono de marca: cercano, claro, dinámico, con un toque
   divertido, sin dejar de ser profesional.
5. No inventar plazos, montos, porcentajes o políticas que no estén
   explícitos en los documentos.

## 5. Guía paso a paso para correr el proyecto

```bash
# 1. Clonar / descomprimir el proyecto
cd agente-ia-bimbam-buy

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env y completar OPENAI_API_KEY o GOOGLE_API_KEY según LLM_PROVIDER

# 5. Generar el índice vectorial (RAG)
python -m src.ingest

# 6. Probar por consola
python main_cli.py

# 7. (Opcional) Probar con interfaz web
streamlit run app_streamlit.py
```

## 6. Preguntas de prueba sugeridas

Usa estas preguntas para validar que el agente responde correctamente y con
citas de fuente:

- "¿Cuánto tiempo tengo para solicitar una devolución?"
- "¿Qué métodos de pago aceptan y son seguros?"
- "Mi producto dejó de funcionar al mes, ¿qué cobertura tengo?"
- "¿Cuánto cuesta el envío y cuánto demora?"
- "¿Cómo me hago afiliado y cómo se calculan mis comisiones?"
- "¿Puedo comprar un dron con puntos de fidelidad?" *(pregunta trampa: no
  está en la base de conocimiento; el agente debe decir que no tiene esa
  información en lugar de inventar).*

## 7. Checklist de entrega (rúbrica sugerida)

- [ ] El agente responde usando **solo** la base de conocimiento entregada.
- [ ] El agente indica la fuente documental de cada respuesta.
- [ ] El agente reconoce cuando no tiene información suficiente.
- [ ] El tono de las respuestas es consistente con la identidad de marca.
- [ ] El proyecto corre de punta a punta siguiendo el `README.md`.
- [ ] Existe al menos una interfaz de usuario funcional (CLI o web).
- [ ] El código está organizado en módulos reutilizables (`src/`).

## 8. Posibles extensiones (para ir más allá)

- Agregar un **re-ranking** de los documentos recuperados antes de pasarlos
  al LLM.
- Sumar una herramienta de **escalamiento a humano** (ej. genera un ticket
  simulado cuando detecta frustración o falta de información).
- Guardar el historial de conversaciones en una base de datos (SQLite) para
  analítica de soporte.
- Desplegar la app de Streamlit en un servicio cloud (Streamlit Cloud,
  Render, etc.).
- Agregar evaluación automática tipo *RAG-eval* (faithfulness / relevancy)
  con un set de preguntas y respuestas esperadas.
