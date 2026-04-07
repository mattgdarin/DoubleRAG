import re
import anthropic
import chromadb

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Answer the query using only the provided context.
If the context does not contain enough information to answer, say so."""

CHUNK_SIZE = 300   # words
CHUNK_OVERLAP = 50 # words
TOP_K = 5


@dataclass
class VanillaRAGResponse:
    answer: str
    sources: list[str] = field(default_factory=list)


class VanillaRAG:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        collection_name: str = "vanilla_rag",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model_name
        self._system_prompt = system_prompt
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(collection_name)

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self._chunk_size - self._chunk_overlap
        return chunks

    def _read_file(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return file_path.read_text(encoding="utf-8")

    def add_file(self, file_path: str) -> None:
        path = Path(file_path)
        text = self._read_file(path)
        chunks = self._chunk_text(text)
        self._collection.add(
            documents=chunks,
            ids=[f"{path.name}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": str(path)} for _ in chunks],
        )

    def add_dir(self, dir_path: str) -> None:
        for path in sorted(Path(dir_path).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
                self.add_file(str(path))

    def respond(self, query: str) -> VanillaRAGResponse:
        results = self._collection.query(query_texts=[query], n_results=self._top_k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = "\n\n---\n\n".join(documents)
        sources = list(dict.fromkeys(m["source"] for m in metadatas))

        response = self._client.messages.create(
            model=self._model,
            system=self._system_prompt,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}],
            max_tokens=4096,
        )
        return VanillaRAGResponse(answer=response.content[0].text, sources=sources)
