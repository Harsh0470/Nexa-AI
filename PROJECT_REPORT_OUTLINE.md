# Minor Project Report Outline

## 1. Title
Offline NLP AI Assistant using Retrieval-Augmented Generation and Kimi

## 2. Abstract
The project develops a document-based AI assistant capable of extracting information
from PDF, DOCX and TXT documents, converting text into semantic embeddings, storing
those embeddings in FAISS, retrieving relevant passages and generating answers with
Kimi.

## 3. Problem Statement
Students and professionals often need to search many notes and documents manually.
Keyword search can miss semantically related information. The proposed system provides
natural-language document question answering.

## 4. Objectives
- Upload common document formats.
- Extract and preprocess text.
- Create semantic vector representations.
- Retrieve relevant document chunks.
- Generate grounded answers.
- Provide a simple Streamlit interface.
- Keep document processing and storage local.

## 5. Technologies
Python, Streamlit, NLTK, Sentence Transformers, FAISS, PyPDF, python-docx and Kimi.

## 6. Methodology
Document ingestion → text extraction → NLP preprocessing → chunking → embeddings →
FAISS indexing → query embedding → similarity retrieval → Kimi generation.

## 7. Expected Outcome
A working AI assistant that can answer questions from a user's uploaded study or
research documents and show the source chunks used for retrieval.

## 8. Limitations
The Kimi API deployment requires internet access. Fully offline Kimi generation
requires a locally deployable Kimi model and suitable hardware.

## 9. Future Scope
OCR, voice assistant, reranking, multilingual support, evaluation framework, local
model deployment and mobile/desktop packaging.
