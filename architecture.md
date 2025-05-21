# FinKnow Architecture

## System Overview

```mermaid
graph TB
    subgraph Frontend
        UI[Streamlit UI]
        subgraph Pages
            Converse[Converse Page<br/>RAG-enabled Chat]
            Chat[Direct Chat<br/>No RAG]
            AddDocs[Add Documents<br/>Document Processing]
        end
    end

    subgraph Backend
        subgraph Document Processing
            Ingestion[Document Ingestion<br/>Text Extraction & Chunking]
            Embedding[OpenAI Embeddings<br/>text-embedding-3-small]
        end

        subgraph Vector Store
            Qdrant[Qdrant Cloud<br/>Vector Database]
        end

        subgraph LLM
            Groq[Groq API<br/>llama3-8b-8192]
        end
    end

    %% Frontend Connections
    UI --> Pages
    Pages --> Converse
    Pages --> Chat
    Pages --> AddDocs

    %% Backend Connections
    AddDocs --> Ingestion
    Ingestion --> Embedding
    Embedding --> Qdrant

    %% RAG Flow
    Converse --> Qdrant
    Qdrant --> Groq

    %% Direct Chat Flow
    Chat --> Groq

    %% Document Processing Flow
    AddDocs --> Ingestion

    classDef frontend fill:#f9f,stroke:#333,stroke-width:2px
    classDef backend fill:#bbf,stroke:#333,stroke-width:2px
    classDef processing fill:#bfb,stroke:#333,stroke-width:2px
    classDef storage fill:#fbb,stroke:#333,stroke-width:2px
    classDef llm fill:#fbf,stroke:#333,stroke-width:2px

    class UI,Pages frontend
    class Ingestion,Embedding processing
    class Qdrant storage
    class Groq llm
```

## Technology Stack

### Frontend
- **Streamlit**: Web application framework for creating interactive UIs
- **Python**: Core programming language

### Backend
- **Document Processing**:
  - Custom text extraction and chunking
  - OpenAI Embeddings API (text-embedding-3-small)

### Vector Store
- **Qdrant Cloud**: Vector database for storing and retrieving document embeddings

### LLM
- **Groq API**: High-performance LLM inference
  - Model: llama3-8b-8192
  - Used for both RAG and direct chat

### Key Features
1. **RAG-enabled Chat**: Document-aware conversations using vector search
2. **Direct Chat**: Pure LLM interactions without document context
3. **Document Management**: Upload and process documents for knowledge base
4. **Vector Search**: Efficient semantic search over document chunks

### Data Flow
1. **Document Processing**:
   - Documents are uploaded through the UI
   - Text is extracted and chunked
   - Chunks are embedded using OpenAI's embedding model
   - Embeddings are stored in Qdrant

2. **RAG Chat**:
   - User question is embedded
   - Similar chunks are retrieved from Qdrant
   - Context and question are sent to Groq LLM
   - Response is generated and displayed

3. **Direct Chat**:
   - User message is sent directly to Groq LLM
   - Response is generated without document context 