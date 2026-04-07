import json
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from agent.base import Agent
from agent.ingestion import IngestionAgent


DEFAULT_SYSTEM_PROMPT = '''You are a research assistant navigating a structured knowledge base.

The knowledge base is organized as topics and children, each containing markdown files.
You will be given an index of available topics and children, and tools to read specific files.

When answering a query:
1. Read the index to identify relevant topics and children.
2. Read the files most likely to contain the answer.
3. If needed, read additional files to fill gaps.
4. Answer concisely, citing the source files you used.
'''


@dataclass
class RAGResponse:
    answer: str
    sources: list[str] = field(default_factory=list)


class RAGAgent(Agent):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        knowledge_dir: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat_history: Optional[list] = None,
        ingestion_model: Optional[str] = None,
        ingestion_system_prompt: Optional[str] = None,
    ):
        super().__init__(api_key, model_name, system_prompt, chat_history)
        self._knowledge_dir = Path(knowledge_dir)
        self._index_path = self._knowledge_dir / ".index.yaml"
        self._ingestion_agent = IngestionAgent(
            api_key=api_key,
            model_name=ingestion_model or model_name,
            knowledge_dir=knowledge_dir,
            **({"system_prompt": ingestion_system_prompt} if ingestion_system_prompt else {}),
        )

    def add_file(self, file_path: str) -> None:
        self._ingestion_agent.ingest(file_path)

    def add_dir(self, dir_path: str) -> None:
        for path in sorted(Path(dir_path).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
                self._ingestion_agent.ingest(str(path))

    def _get_context(self, query: str) -> tuple[str, list[str]]:
        # Step 1: load index
        index = yaml.safe_load(self._index_path.read_text())
        topics = index.get("topics", {})

        # Step 2: find relevant topics
        topic_message = {"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Available topics: {list(topics.keys())}\n\n"
            "Which topics are relevant to this query? Reply with a JSON array of topic keys, e.g. [\"machine_learning\"]. "
            "If none are relevant, reply with []."
        )}
        relevant_topics = json.loads(self._send([topic_message]))

        # Step 3: find relevant children per topic
        relevant_children: dict[str, list[str]] = {}
        for topic_key in relevant_topics:
            children = topics[topic_key].get("children", {})
            child_message = {"role": "user", "content": (
                f"Query: {query}\n\n"
                f"Topic: {topic_key}\n"
                f"Available children: {list(children.keys())}\n\n"
                "Which children are relevant to this query? Reply with a JSON array of child keys. "
                "If none are relevant, reply with []."
            )}
            relevant_children[topic_key] = json.loads(self._send([child_message]))

        # Step 4: read files
        context_parts: list[str] = []
        sources: list[str] = []
        for topic_key, child_keys in relevant_children.items():
            for child_key in child_keys:
                child_dir = self._knowledge_dir / topic_key / child_key
                for file in sorted(child_dir.glob("*.md")):
                    context_parts.append(file.read_text(encoding="utf-8"))
                    sources.append(str(file.relative_to(self._knowledge_dir)))

        return "\n\n---\n\n".join(context_parts), sources

    def respond(self, query: str) -> RAGResponse:
        context, sources = self._get_context(query)
        answer = self._send([{"role": "user", "content": (
            f"Context:\n{context}\n\n"
            f"Query: {query}"
        )}])
        return RAGResponse(answer=answer, sources=sources)
        
