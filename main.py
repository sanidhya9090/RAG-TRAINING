import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("NVIDIA API key not found. Set NVIDIA_API_KEY in the .env file.")

# ---------------------------------------------------------------------------
# 2. Locate the PDF
#    Works with the full book or the 200-page extract.
# ---------------------------------------------------------------------------
BOOK_TITLE = "Operating System Concepts (10th Edition) by Silberschatz, Galvin and Gagne"

base_dir = Path(__file__).resolve().parent
pdf_candidates = [
    base_dir / "Abraham-Silberschatz-Operating-System-Concepts-10th-2018.pdf",
    base_dir / "Operating-System-Concepts-first-200-pages.pdf",
]
pdf_path = next((p for p in pdf_candidates if p.exists()), None)
if pdf_path is None:
    matches = list(base_dir.glob("*Operating*System*Concepts*.pdf"))
    if not matches:
        raise FileNotFoundError(
            f"No Operating System Concepts PDF found in {base_dir}. "
            "Add the PDF or update pdf_candidates."
        )
    pdf_path = matches[0]

# ---------------------------------------------------------------------------
# 3. Embeddings + persistent Qdrant store
#    The index is saved to disk, so the PDF is only embedded once.
#    (Delete the ./qdrant_db folder if you switch to a different PDF.)
# ---------------------------------------------------------------------------
COLLECTION = "os_concepts"
DB_PATH = str(base_dir / "qdrant_db")

embeddings = FastEmbedEmbeddings()  # default model: 384-dim
client = QdrantClient(path=DB_PATH)

needs_indexing = (
    not client.collection_exists(COLLECTION)
    or client.count(COLLECTION).count == 0
)

if needs_indexing:
    print(f"Loading document from: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path)).load()
    print(f"Loaded {len(docs)} pages.")

    # Textbook prose and code listings need larger chunks than 600 chars
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Drop near-empty chunks (page numbers, blank pages, figure-only pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    print(f"Created {len(chunks)} text chunks.")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION, embedding=embeddings
    )

    print("Embedding and indexing (first run only, this can take a few minutes)...")
    BATCH = 128
    for i in range(0, len(chunks), BATCH):
        vectorstore.add_documents(chunks[i : i + BATCH])
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")
    print("Indexing complete!")
else:
    print("Found existing index, skipping indexing.")
    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION, embedding=embeddings
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------------------------------------------------------------------------
# 4. LLM
# ---------------------------------------------------------------------------
# nemotron uses reasoning tokens, so keep max_completion_tokens high
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

# ---------------------------------------------------------------------------
# 5. Prompts
# ---------------------------------------------------------------------------
# Rewrites follow-ups ("what about its disadvantages?") into standalone
# questions so retrieval works across a multi-turn conversation.
condense_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest user question, rewrite the "
        "question so it is fully self-contained and can be understood without "
        "the history. Do NOT answer it. If it is already self-contained, "
        "return it unchanged. Output only the rewritten question.",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"You are an expert Operating Systems tutor using the textbook "
        f"'{BOOK_TITLE}'. "
        "Answer the question accurately based ONLY on the provided context. "
        "Explain clearly, and when helpful use short examples, step-by-step "
        "reasoning, or code taken from the context. "
        "If the answer cannot be found in the context, say you don't know "
        "rather than guessing.\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

condense_chain = condense_prompt | llm | StrOutputParser()
qa_chain = qa_prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# 6. Chat logic
# ---------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 6  # last 3 exchanges


def format_context(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
    )


def chat(question: str, history: list):
    recent = history[-MAX_HISTORY_MESSAGES:]

    # Make follow-up questions searchable
    search_query = question
    if recent:
        search_query = condense_chain.invoke(
            {"history": recent, "question": question}
        ).strip() or question

    docs = retriever.invoke(search_query)
    answer = qa_chain.invoke(
        {
            "context": format_context(docs),
            "history": recent,
            "question": question,
        }
    )

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
    return answer, pages, search_query


def main():
    print("\n" + "=" * 60)
    print("Operating System Concepts chatbot")
    print("Commands: 'exit' to quit, 'clear' to reset the conversation")
    print("=" * 60)

    history = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if question.lower() == "clear":
            history.clear()
            print("Conversation cleared.")
            continue

        try:
            answer, pages, search_query = chat(question, history)
        except Exception as e:
            print(f"\nError: {e}")
            continue

        print(f"\nTutor: {answer}")
        print(f"\n(Sources: PDF pages {', '.join(map(str, pages))})")


if __name__ == "__main__":
    main()