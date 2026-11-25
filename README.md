# Operating System Concepts RAG Chatbot

A command-line study assistant that answers questions about the textbook *Operating System Concepts (10th Edition)* by Silberschatz, Galvin and Gagne. It uses **Retrieval-Augmented Generation (RAG)**: your question is matched against passages from the PDF, and an LLM answers using only those passages, citing the PDF pages it used.

## Features

- Answers grounded in the textbook, with page references for every answer
- Multi-turn conversation: follow-ups like "what about its disadvantages?" are rewritten into standalone questions before searching
- Persistent local vector index (Qdrant), so the PDF is embedded only once
- Local embeddings (FastEmbed), so only the answer generation uses an API
- Says "I don't know" instead of guessing when the answer isn't in the book

## How it works

```
PDF ──► PyPDFLoader ──► text splitter (1000 chars, 150 overlap)
                              │
                     FastEmbed (384-dim vectors)
                              │
                      Qdrant (saved in ./qdrant_db)
                              │
 question ──► condense follow-up ──► retrieve top 5 chunks
                                          │
                       NVIDIA-hosted LLM answers from the chunks
                                          │
                                answer + source pages
```

## Project structure

```
os-rag-chatbot/
├── main.py                 # The chatbot
├── requirements.txt        # Python dependencies
├── .env.example            # Template for your API key
├── .env                    # Your real key (git-ignored, create it yourself)
├── .gitignore
├── README.md
├── Operating-System-Concepts-first-200-pages.pdf   # Your textbook PDF (you supply this)
└── qdrant_db/              # Created automatically on first run
```

## Prerequisites

- Python 3.10 or newer
- An NVIDIA API key from <https://build.nvidia.com/> (free tier available)
- The textbook PDF placed next to `main.py`

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # macOS / Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key**

   ```bash
   cp .env.example .env        # Windows: copy .env.example .env
   ```

   Edit `.env`:

   ```
   NVIDIA_API_KEY=your_nvidia_api_key_here
   ```

4. **Add the PDF** next to `main.py`. The script looks for these, in order:
   1. `Abraham-Silberschatz-Operating-System-Concepts-10th-2018.pdf` (full book)
   2. `Operating-System-Concepts-first-200-pages.pdf` (extract)
   3. Any PDF whose name matches `*Operating*System*Concepts*.pdf`

## Usage

```bash
python main.py
```

The first run loads the PDF, splits it into chunks and builds the index. This takes a few minutes the first time and is skipped on later runs.

```
You: What is the difference between a process and a thread?

Tutor: A process is a program in execution with its own address space... 

(Sources: PDF pages 120, 121, 125)
```

Commands: `clear` resets the conversation, `exit` (or `quit`, `q`) leaves.

## Configuration

Edit these values in `main.py`:

| Setting | Default | Purpose |
|---|---|---|
| `chunk_size` / `chunk_overlap` | 1000 / 150 | Size of text pieces stored in the index |
| `search_kwargs={"k": ...}` | 5 | Number of passages retrieved per question |
| `MAX_HISTORY_MESSAGES` | 6 | How many recent messages are remembered (3 exchanges) |
| `model` | `nvidia/nemotron-3.5-lightning-30b-a3b` | LLM used to answer |
| `max_completion_tokens` | 2500 | Keep high: this model spends tokens on reasoning |
| `temperature` | 0.2 | Lower is more factual |

## Switching to a different PDF

The index is tied to the PDF it was built from. If you change the PDF (for example from the 200-page extract to the full book), delete the `qdrant_db/` folder and run again.

If you change the embedding model, also update `size=384` in `main.py` to match its vector dimension, then delete `qdrant_db/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `NVIDIA API key not found` | Create `.env` (with the leading dot) in the folder you run from and set `NVIDIA_API_KEY`. |
| `No Operating System Concepts PDF found` | Put the PDF next to `main.py` or add its name to `pdf_candidates`. |
| Answers come from the wrong book or are missing pages | Delete `qdrant_db/` so the index is rebuilt from the current PDF. |
| `Storage folder ... is already accessed by another instance` | Only one process can open the local Qdrant folder at a time. Close any other running copy. |
| Model not found / 404 | NVIDIA occasionally renames models. Check <https://build.nvidia.com/models> and update `model=` in `main.py`. |
| Empty or odd answers | Raise `max_completion_tokens`; reasoning models can use up the limit before answering. |
| Scanned PDF returns no text | `PyPDFLoader` reads text only. Run OCR on the PDF first. |

## Security notes

- Never commit `.env`. It is listed in `.gitignore`.
- If your real key was ever shared publicly, revoke it and generate a new one.
- The textbook is copyrighted. Keep the PDF out of public repositories (it is git-ignored here).

## Ideas for next steps

- Stream answers token by token
- Add hybrid search (dense + BM25) for better keyword matching
- Wrap the chat function in a FastAPI endpoint or a Streamlit/Gradio UI
- Add a reranker to improve which passages reach the LLM
