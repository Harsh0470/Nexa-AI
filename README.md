# Nexa AI — Offline NLP Document Assistant (Kimi)

Nexa AI is a student-friendly **Retrieval-Augmented Generation (RAG)** project. It lets you upload PDF, DOCX and TXT documents, build a local FAISS knowledge base, and ask natural-language questions about those documents.

> **Important:** document processing, embeddings and FAISS retrieval run locally. The final question and retrieved context are sent to the configured Kimi/Moonshot API. Therefore the default API setup is not fully offline.

## What's new in this version

- Redesigned modern Streamlit interface
- Cleaner sidebar with separate Knowledge Base, Workspace and System Status sections
- Dashboard cards for document count, index status and conversation size
- New conversation button
- Improved empty-state onboarding
- Cleaner source display with retrieved chunk count
- Same PDF/DOCX/TXT → NLP → embeddings → FAISS → Kimi workflow
- Same `.env` configuration and project structure

## Features

- PDF / DOCX / TXT upload
- Local text extraction
- NLTK preprocessing
- Sentence-transformer embeddings
- FAISS semantic search
- Kimi answer generation
- Source/chunk display
- Persistent local document/index folders
- Streamlit web UI
- No LangChain dependency

## Folder structure

```text
offline_nlp_ai_assistant_kimi/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── PROJECT_REPORT_OUTLINE.md
├── utils/
│   ├── document_loader.py
│   ├── text_processor.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── kimi_client.py
└── data/
    ├── documents/
    └── index/
```

## Windows setup

Open the project folder in VS Code Terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Kimi API key:

```text
KIMI_API_KEY=your_key
```

Then run:

```powershell
streamlit run app.py
```

## How the pipeline works

```text
Document
   ↓
PDF/DOCX/TXT extraction
   ↓
NLP cleaning
   ↓
Chunking
   ↓
Sentence Transformer embeddings
   ↓
FAISS vector index
   ↓
User question
   ↓
Question embedding
   ↓
Top-k similarity search
   ↓
Retrieved context
   ↓
Kimi
   ↓
Grounded answer + sources
```

## Viva explanation

**Why FAISS?**
It stores numerical embeddings and performs fast similarity search.

**Why embeddings?**
Embeddings represent text as vectors, allowing semantically similar content to be retrieved even when the exact keywords differ.

**Why RAG?**
RAG retrieves relevant parts of the user's documents and gives them to the language model as context instead of relying only on the model's general knowledge.

**Why Kimi?**
Kimi acts as the answer-generation layer after retrieval. The generation endpoint remains configurable through `kimi_client.py`.

**Why Streamlit?**
It provides a simple Python-based web interface without requiring a separate frontend application.
