#just to fix import errors in retriever.py


import re
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DB_DIR = SRC_DIR / "chroma_store"


def load_transcripts():
    files = [f for f in DATA_DIR.iterdir() if f.suffix.lower() in {".vtt", ".srt", ".txt"}]
    if not files:
        raise FileNotFoundError(f"No transcript files in {DATA_DIR}")

    docs = []
    for path in files:
        lines = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if (not line or line == "WEBVTT" or "-->" in line
                    or line.isdigit() or line.startswith(("NOTE", "Kind:", "Language:"))):
                continue
            line = re.sub(r"<[^>]+>", "", line)
            if lines and lines[-1] == line:
                continue
            lines.append(line)

        match = re.search(r"Session\s*(\d+)", path.name, re.IGNORECASE)
        session = match.group(1) if match else "unknown"
        docs.append(Document(page_content=" ".join(lines),
                             metadata={"session": session, "source": path.name}))
    return docs


def load_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        store = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
        if store._collection.count() > 0:
            return store

    docs = load_transcripts()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=768, chunk_overlap=150
    ).split_documents(docs)
    return Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
if __name__ == "__main__":
    store = load_store()
    print("Chunks in store:", store._collection.count())

    retriever = store.as_retriever(search_kwargs={"k": 5})
    query = "what is regression testing?"
    for r in retriever.invoke(query):
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")