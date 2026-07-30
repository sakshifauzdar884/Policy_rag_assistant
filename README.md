Policy RAG Assistant

Overview: Policy RAG Assistant is an AI-powered question-answering system built using Python. It helps employees quickly find accurate information from company policy documents such as Expense Policy, Travel Policy, Finance Policy, and Employee Handbook.The application uses Retrieval-Augmented Generation (RAG), which retrieves the most relevant information from the provided documents before generating an answer. Instead of guessing or creating false information, the assistant only answers using the available documents and provides proper citations for every response. If the requested information is not found, the assistant clearly informs the user that the answer is unavailable.This project demonstrates document retrieval, hybrid search, citation generation, query rewriting, conversation history, and confidence-based response generation.

Features:
- AI-powered Policy Assistant
- Retrieval-Augmented Generation (RAG)
- Hybrid Search using TF-IDF and BM25
- Source Citation Support
- Multi-turn Conversation
- Query Rewriting
- Metadata Filtering
- Incremental Document Indexing
- Confidence-based "I don't know" Responses
- Optional Claude LLM Integration
- Works without an API Key
- Automated Test Cases

Technologies Used:
- Python
- Scikit-learn
- Flask
- TF-IDF
- BM25
- Anthropic Claude (Optional)
- Markdown Documents
- Pickle
- PyTest

How the System Works: The application follows the Retrieval-Augmented Generation (RAG) workflow-
1. Load policy documents.
2. Split documents into meaningful chunks.
3. Build TF-IDF and BM25 indexes.
4. Accept the user's question.
5. Rewrite follow-up questions if required.
6. Retrieve the most relevant document sections.
7. Apply metadata filtering and reranking.
8. Generate the final response.
9. Add source citations.
10. Return the answer to the user.

Installation:

## Clone Repository

```bash
git clone <repository-url>
cd Policy-RAG-Assistant
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Automated Tests

```bash
python -m pytest tests/ -v
```

## Run Demo

```bash
python run_demo.py
```

## Start Interactive Chat

```bash
python -m src.cli
```

## Rebuild Document Index

```bash
python -m src.cli --rebuild
```

---

# Architecture

```text
Policy Documents
        │
        ▼
Document Loader
        │
        ▼
Document Chunking
        │
        ▼
Indexing (TF-IDF + BM25)
        │
        ▼
Hybrid Retriever
        │
        ▼
Answer Generator
        │
        ▼
Response with Citations
```
Sample Questions
- What is the meal reimbursement limit?
- What is the travel policy for international trips?
- How many annual leaves are allowed?
- Can I claim gym membership expenses?
- What is the purchase approval process?

 Design Decisions:
- Header-aware chunking improves retrieval accuracy.
- Hybrid Search combines TF-IDF and BM25.
- Metadata filtering improves search relevance.
- Query expansion improves retrieval quality.
- Confidence scoring prevents hallucinations.
- Conversation history enables follow-up questions.
- Supports both LLM-based and extractive responses.

Assumptions:
- Sample policy documents are included because the original documents were not provided.
- The system works with any compatible Markdown or text policy files.
- Currency values are examples only.
- Out-of-scope questions return an appropriate "not found" response.
- The application assumes a single user chat session.

Trade-offs:
- TF-IDF and BM25 are lightweight and easy to deploy but less powerful than dense embeddings for semantic matching.
- Incremental indexing rebuilds lightweight statistics instead of performing true online updates.
- Confidence thresholds are heuristic and may require tuning for larger datasets.
- Without an LLM API key, the assistant provides extractive responses instead of generated summaries.
- OCR support for scanned PDFs is not implemented.


Future Improvements:
- Add Sentence Transformer embeddings
- Cross-Encoder Reranking
- OCR Support for Scanned PDFs
- Streaming LLM Responses
- User Feedback Collection
- Pinecone / Weaviate / pgVector Integration
- Role-Based Access Control
- Web-based User Interface
- Docker Deployment
- Cloud Hosting

Why Retrieval-Augmented Generation (RAG)?
Traditional language models may generate incorrect or outdated information. RAG improves answer quality by first retrieving relevant information from trusted documents and then generating responses based only on those documents. This reduces hallucinations and ensures every answer is supported with proper citations.

Key Highlights:
- Hybrid Retrieval System
- Explainable AI Responses
- Source Citations
- Conversation Memory
- Incremental Indexing
- Metadata Filtering
- LLM Optional
- Production-Oriented Architecture



Author
Sakshi Fauzdar

