import re
import yaml
from .base import Agent
from pathlib import Path
from typing import Optional

DEFAULT_SYSTEM_PROMPT = '''You are a knowledge base librarian. Your job is to integrate new documents into an existing knowledge hierarchy.

The knowledge base is a directory of markdown files organized into topic folders. Each file has YAML frontmatter with: id, parent, tags, and merged_from (a list of source files whose content was merged here). A single index.yaml tracks the full hierarchy.

When given a new file, you must:
1. Read the existing index.yaml to understand the current structure.
2. Identify where the new content belongs — existing topic, new topic, or new subtopic.
3. Check for overlap with existing files. If significant overlap exists, merge the content rather than creating a duplicate. Preserve all unique information from both.
4. Write clean, well-structured markdown. Remove formatting artifacts, redundancy, and noise from the source. Keep only the knowledge.
5. Update index.yaml to reflect any new or modified files.

Principles:
- One idea, one place. Never let the same concept live in two files.
- Merge aggressively. A smaller, denser knowledge base is better than a large, redundant one.
- Preserve provenance. Always record source files in merged_from.
- Hierarchy should reflect concepts, not the structure of the original source files.
'''


class IngestionAgent(Agent):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        knowledge_dir: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat_history: Optional[list] = None,
        chunk_size: int = 1500,
        chunk_break: Optional[list[str]] = None,
        chunk_overlap: Optional[int] = None,
    ):
        super().__init__(api_key, model_name, system_prompt, chat_history)
        self._knowledge_dir = Path(knowledge_dir)
        self._index_path = self._knowledge_dir / ".index.yaml"
        self._chunk_size = chunk_size
        self._chunk_breaks = chunk_break if chunk_break is not None else ["\n\n"]
        self._chunk_overlap = chunk_overlap
        self._setup()

    def _setup(self) -> None:
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._index_path.write_text("topics: {}\n")


    def _chunk_text(self, text: str) -> list[str]:
        pattern = "|".join(re.escape(b) for b in self._chunk_breaks)
        paragraphs = [p.strip() for p in re.split(pattern, text) if p.strip()]
        chunks = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para.split())

            if para_len > self._chunk_size:
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_len = [], 0
                sentences = para.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    s_len = len(sentence.split())
                    if current_len + s_len > self._chunk_size and current:
                        chunks.append("\n\n".join(current))
                        current, current_len = [], 0
                    current.append(sentence)
                    current_len += s_len
                continue

            if current_len + para_len > self._chunk_size and current:
                chunks.append("\n\n".join(current))
                if self._chunk_overlap:
                    tail = " ".join("\n\n".join(current).split()[-self._chunk_overlap:])
                    current, current_len = [tail], len(tail.split())
                else:
                    current, current_len = [], 0

            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _clean_key(self, raw: str) -> str:
        return raw.strip().strip("`").strip()

    def _send_with_retry(self, messages: list, valid_keys: set, label: str) -> str:
        for attempt in range(2):
            raw = self._send(messages).strip()
            key = self._clean_key(raw)
            if key == "none" or key in valid_keys:
                return key
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        f"'{key}' is not a valid option. "
                        f"Valid options are: {sorted(valid_keys)} or 'none'. "
                        "Reply with just the key, no backticks or formatting."
                    )},
                ]
        print(f"[warning] could not resolve {label} after 2 attempts, skipping chunk")
        return "none"

    def _read_file(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return file_path.read_text(encoding="utf-8")

    def ingest(self, file_path: str) -> None:
        source_name = Path(file_path).name
        text = self._read_file(Path(file_path))
        chunks = self._chunk_text(text)
        index = yaml.safe_load(self._index_path.read_text())
        for chunk in chunks:
            # --- Step 1: find topic ---
            topic_message = {"role": "user", "content": (
                f"Given this list of existing topics: {list(index['topics'].keys())}\n\n"
                f"And this chunk of text:\n{chunk}\n\n"
                "Which single topic is this chunk most relevant to? Reply with just the topic key, no backticks or formatting. "
                "If none are sufficiently relevant, reply with 'none'."
            )}
            topic_key = self._send_with_retry([topic_message], set(index["topics"].keys()), "topic")

            if topic_key == "none":
                new_topic_prompt = (
                    f"None of the existing topics fit this chunk. "
                    f"Existing topics: {list(index['topics'].keys()) or 'none yet'}. "
                    "Propose a concise topic name (2-4 words) that fits alongside them. "
                    "Reply with just the name, nothing else."
                )
                topic_label = self._send([
                    topic_message,
                    {"role": "assistant", "content": "none"},
                    {"role": "user", "content": new_topic_prompt},
                ]).strip()
                topic_key = re.sub(r"[^a-z0-9]+", "_", topic_label.lower()).strip("_")
                (self._knowledge_dir / topic_key).mkdir(parents=True, exist_ok=True)
                index["topics"][topic_key] = {"label": topic_label, "children": {}}

            # --- Step 2: find child ---
            children = index["topics"][topic_key].get("children", {})
            child_message = {"role": "user", "content": (
                f"Topic: {topic_key}\n"
                f"Existing children: {list(children.keys())}\n\n"
                f"Chunk:\n{chunk}\n\n"
                "Which single child is this chunk most relevant to? Reply with just the child key, no backticks or formatting. "
                "If none are sufficiently relevant, reply with 'none'."
            )}
            child_key = self._send_with_retry([child_message], set(children.keys()), "child")

            if child_key == "none":
                new_child_prompt = (
                    f"None of the existing children fit this chunk. "
                    f"Existing children: {list(children.keys()) or 'none yet'}. "
                    "Propose a concise child name (2-4 words) that fits under the topic. "
                    "Reply with just the name, nothing else."
                )
                child_label = self._send([
                    child_message,
                    {"role": "assistant", "content": "none"},
                    {"role": "user", "content": new_child_prompt},
                ]).strip()
                child_key = re.sub(r"[^a-z0-9]+", "_", child_label.lower()).strip("_")
                index["topics"][topic_key]["children"][child_key] = {
                    "label": child_label,
                }

            # --- Step 3: merge or create ---
            child_dir = self._knowledge_dir / topic_key / child_key
            child_dir.mkdir(parents=True, exist_ok=True)
            existing_files = list(child_dir.glob("*.md"))

            if existing_files:
                summaries = "\n".join(
                    f"{f.name}:\n{f.read_text(encoding='utf-8')}" for f in existing_files
                )
                merge_message = {"role": "user", "content": (
                    f"These files exist under {topic_key}/{child_key}:\n{summaries}\n\n"
                    f"New chunk:\n{chunk}\n\n"
                    "Which file should this chunk be merged into? Reply with just the filename (e.g. neural_networks.md). "
                    "If it does not belong in any of them, reply with 'none'."
                )}
                merge_target = self._send([merge_message]).strip()

                if merge_target != "none":
                    target_file = child_dir / merge_target
                    merged = self._send([
                        merge_message,
                        {"role": "assistant", "content": merge_target},
                        {"role": "user", "content": (
                            "Rewrite the file incorporating the new chunk. "
                            "Eliminate redundancy. Preserve all unique information. "
                            "Keep the existing YAML frontmatter, updating merged_from if needed. "
                            f"Where you incorporate content from the new chunk, append an inline citation: *(Source: {source_name})*. "
                            "Reply with only the file contents."
                        )},
                    ])
                    target_file.write_text(merged, encoding="utf-8")
                else:
                    file_label = self._send([
                        merge_message,
                        {"role": "assistant", "content": "none"},
                        {"role": "user", "content": (
                            "Propose a concise filename (2-4 words, no extension) for this chunk. "
                            "Reply with just the name."
                        )},
                    ]).strip()
                    file_key = re.sub(r"[^a-z0-9]+", "_", file_label.lower()).strip("_")
                    (child_dir / f"{file_key}.md").write_text(chunk + f"\n\n*(Source: {source_name})*", encoding="utf-8")
            else:
                file_label = self._send([{"role": "user", "content": (
                    f"Propose a concise filename (2-4 words, no extension) for this chunk:\n{chunk}\n\n"
                    "Reply with just the name."
                )}]).strip()
                file_key = re.sub(r"[^a-z0-9]+", "_", file_label.lower()).strip("_")
                (child_dir / f"{file_key}.md").write_text(chunk + f"\n\n*(Source: {source_name})*", encoding="utf-8")

        self._index_path.write_text(yaml.dump(index, allow_unicode=True))

