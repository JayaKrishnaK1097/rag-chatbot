# RAG Chatbot

A chatbot that answers questions about a PDF document using AI.

Ask it anything about the document and it gives you a grounded answer based only on the content — not made-up information. Built with Google's Gemini and Python.

## How it works

1. The PDF is split into small chunks and stored as vectors in a local database
2. When you ask a question, the system finds the most relevant chunks
3. Those chunks plus your question are sent to Gemini, which writes the answer

## Tech stack

- **Python** — main language
- **Google Gemini** — the AI model (free tier)
- **LangChain** — connects the pieces together
- **ChromaDB** — local vector database
- **FastAPI** — REST API framework

## Setup

You need Python 3.11+ and a free Google AI Studio API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/JayaKrishnaK1097/rag-chatbot.git
cd rag-chatbot
python -m venv venv
venv\Scripts\activate

pip install langchain langchain-google-genai langchain-chroma langchain-classic langchain-text-splitters langchain-community fastapi uvicorn pypdf python-dotenv
```

Create a `.env` file with your API key:

GOOGLE_API_KEY=your-key-here(Build your own key!!!)

## Run it

```bash
# One time — load the PDF into the database
python ingest.py

# Start the API
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser to test it.

## Files

- `ingest.py` — reads the PDF and stores it
- `rag_chain.py` — the retrieval + AI logic
- `main.py` — the API endpoints
- `Resume.pdf` — sample document

## Example

Ask: *"What programming languages does Jaya know?"*

Get: *"Jaya knows C#, Python, Java, JavaScript, and SQL..."*

## What's next

- Add streaming responses
- Support multiple documents
- Deploy to the cloud