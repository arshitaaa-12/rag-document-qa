# Document Q&A Bot (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about custom documents. The system retrieves relevant context from the provided text and uses a Large Language Model to generate accurate, grounded answers.

## Features

- **Document Ingestion:** Loads text documents and splits them into optimized chunks.
- **Vector Search:** Uses HuggingFace sentence-transformers and ChromaDB for semantic similarity search.
- **LLM Integration:** Uses Groq via LangChain for fast, context-aware generation.
- **Interactive UI:** A clean Streamlit chat interface with stateful memory and a conversation reset button.
- **Terminal Interface:** Includes a command-line version for quick testing.

## Tech Stack

- **Language:** Python
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **LLM Provider:** Groq API
- **Frontend:** Streamlit

## Architecture

1. The document is loaded and split into smaller chunks.
2. Chunks are converted into vector embeddings and stored in ChromaDB.
3. When a user asks a question, the system performs a semantic search to find the most relevant chunks.
4. The retrieved chunks are passed as context to the LLM, which generates a final answer.

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/arshitaaa-12/rag-document-qa.git
   cd rag-document-qa
