# 🎬 YouTube Chatbot using LangChain + Ollama

A local RAG (Retrieval-Augmented Generation) chatbot that lets you query any YouTube video using its transcript — powered by LangChain, FAISS, HuggingFace embeddings, and Ollama (Mistral).

---

## 🧠 How It Works

1. Fetches the transcript of a YouTube video
2. Splits it into chunks and embeds them using `all-MiniLM-L6-v2`
3. Stores embeddings in a FAISS vector store
4. Retrieves relevant chunks based on your query
5. Passes context + question to Mistral (via Ollama) for a grounded answer

---

## 📁 Project Structure
```
yt-chatbot-langchain/
├── ingestion/
│   ├── __init__.py
│   └── youtube_loader.py    # Fetches YouTube transcript
├── processing/
│   ├── __init__.py
│   └── splitter.py          # Splits transcript into chunks
├── vector_store/
│   ├── __init__.py
│   └── faiss_store.py       # Creates FAISS vector store
├── retrieval/
│   ├── __init__.py
│   └── retriever.py         # Retrieves relevant chunks
├── utils/
│   ├── __init__.py
│   └── prompt.py            # Prompt template
├── app.py                   # Main pipeline
├── config.py                # Model config and env loading
├── requirements.txt
└── .env                     # API keys (never committed)
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Synapse-CodeX/Yt-chatbot-using-Langchain.git
cd Yt-chatbot-using-Langchain
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up `.env` file
Create a `.env` file in the root directory:
HUGGINGFACE_API_KEY=your_huggingface_api_key

### 5. Pull the Mistral model via Ollama
```bash
ollama pull mistral
```

### 6. Run
```bash
# Interactive prompt for video ID and questions
python app.py

# Or pass the YouTube video ID directly as an argument
python app.py --video-id 8TZMtslA3UY
```

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Mistral via Ollama (local) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS |
| Transcript | `youtube-transcript-api` |

---

## 📌 Notes

- Ollama must be running locally before executing the script
- The video must have captions/subtitles enabled on YouTube
- The `.env` file is gitignored — never commit your API keys
