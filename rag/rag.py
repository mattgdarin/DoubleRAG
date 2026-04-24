import json
import yaml
import re
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
5. If sources contain conflicting information on the same topic, explicitly note the conflict and cite both sources.
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
        user_preferences: Optional[list] = None,
        ingestion_system_prompt: Optional[str] = None,
    ):
        super().__init__(api_key, model_name, system_prompt, chat_history)
        self._user_preferences: list[str] = list(user_preferences) if user_preferences else []
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
    
    def clear_history(self) -> None:
        self.chat_history = []


    def add_dir(self, dir_path: str) -> None:
        for path in sorted(Path(dir_path).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
                self._ingestion_agent.ingest(str(path))

    def _get_context(self, query: str) -> tuple[str, list[str]]:
        # Step 1: load index
        index = yaml.safe_load(self._index_path.read_text())
        topics = index.get("topics", {})
        recent_history = self._chat_history[-6:]  # last 3 exchanges (6 messages)

        # Step 2: find relevant topics
        topic_message = {"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Available topics: {list(topics.keys())}\n\n"
            "Which topics are relevant to this query? Reply with a JSON array of topic keys, e.g. [\"machine_learning\"]. "
            "If none are relevant, reply with []."
        )}
        try:
            raw = self._send(recent_history + [topic_message])
            raw = re.sub(r"```[\w]*\n?", "", raw).strip()
            relevant_topics = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            print("[warning] could not parse topic response, defaulting to no topics")
            relevant_topics = []

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
            try:
                raw = re.sub(r"```[\w]*\n?", "", self._send(recent_history + [child_message])).strip()
                relevant_children[topic_key] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                print(f"[warning] could not parse child response for {topic_key}, skipping")
                relevant_children[topic_key] = []

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

    def _add_context(self, recent_history: list) -> None:
        user_messages = "\n".join(
            msg['content'] for msg in recent_history if msg['role'] == 'user'
        )
        formatted = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history
        )
        decision = self._send([{"role": "user", "content": (
            f"Here is a recent conversation:\n\n{formatted}\n\n"
            "Does the user (not the assistant) share any factual information or domain knowledge worth saving for future reference? "
            "Use the assistant's messages only for context to interpret what the user is saying. "
            "This includes technical facts, definitions, explanations, or context about the user's project or field. "
            "Do NOT include personal preferences, stylistic choices, or how the user likes to communicate — only objective knowledge. "
            "Reply with 'yes' or 'no', and if yes, briefly summarise the facts worth saving in at most three sentences."
        )}]).strip()

        if decision.lower().startswith("yes"):
            summary = decision[decision.lower().find("yes") + 3:].strip().lstrip(",:").strip()
            print(f"\n[DoubleRAG] This conversation contains potentially useful context: {summary}")
            print("[DoubleRAG] Would you like to save this to the knowledge base? (yes/no): ", end="", flush=True)
            if input().strip().lower() == "yes":
                import tempfile, os
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                    tmp.write(summary)
                    tmp_path = tmp.name
                try:
                    self._ingestion_agent.ingest(tmp_path)
                finally:
                    os.unlink(tmp_path)

        # --- Step 2: extract preferences ---
        pref_decision = self._send([{"role": "user", "content": (
            f"Here are the last few messages from a conversation:\n\n{formatted}\n\n"
            "Does this conversation reveal personal preferences about how the user likes to work or communicate? "
            "This includes things like: preferred response format, level of detail, tone, tools or frameworks they favour, "
            "or how they like explanations structured. "
            "Do NOT include factual or domain knowledge — only subjective preferences. "
            "Reply with 'yes' or 'no', and if yes, state each preference as a short bullet point."
        )}]).strip()

        if pref_decision.lower().startswith("yes"):
            prefs_text = pref_decision[pref_decision.lower().find("yes") + 3:].strip().lstrip(",:").strip()
            new_prefs = [p.strip().lstrip("-•").strip() for p in prefs_text.splitlines() if p.strip()]
            self._user_preferences.extend(new_prefs)
            print(f"\n[DoubleRAG] New preferences detected: {new_prefs}")
            print("[DoubleRAG] Would you like to save these preferences? (yes/no): ", end="", flush=True)
        


    def respond(self, query: str) -> RAGResponse:
        context, sources = self._get_context(query)

        if not context:
            print("[DoubleRAG] Insufficient context in knowledge base. Search the web? (yes/no): ", end="", flush=True)
            if input().strip().lower() == "yes":
                context, sources = self._search_web(query)
                used_web = bool(sources)
            else:
                return RAGResponse(
                    answer="I don't have any information on that topic in the knowledge base.",
                    sources=[],
                )
        else:

            sufficient = self._send([{"role": "user", "content": (
                f"Context:\n{context}\n\n"
                f"Query: {query}\n\n"
                "Does the context contain enough information to answer the query? Reply with only the single word 'yes' or 'no', nothing else."
            )}]).strip().lower().strip("*.,!?")

            used_web = False
            if sufficient != "yes":
                print("[DoubleRAG] Insufficient context in knowledge base. Search the web? (yes/no): ", end="", flush=True)
                if input().strip().lower() == "yes":
                    context, sources = self._search_web(query)
                    used_web = bool(sources)
                else:
                    context = f"[Note: the knowledge base has limited information on this topic.] {context}"
            
        


        prefs_block = (
            "\n\nUser preferences:\n" + "\n".join(f"- {p}" for p in self._user_preferences)
            if self._user_preferences else ""
        )
        user_message = {"role": "user", "content": (
            f"Context:\n{context}\n\n"
            f"Query: {query}"
            f"{prefs_block}"
        )}
        answer = self._stream(self._chat_history + [user_message])
        self._chat_history.append({"role": "user", "content": query})
        self._chat_history.append({"role": "assistant", "content": answer})

        if used_web:
            print(
                "\n[DoubleRAG] Web content was used to answer this query. "
                "Would you like to ingest it into the knowledge base for future use? "
                "Note: ingestion may take a while and be expensive — only recommended if this content will be regularly useful. (yes/no): ",
                end="", flush=True
            )
            if input().strip().lower() == "yes":
                import tempfile, os
                web_sources = [s[len("[web]"):].strip() for s in sources if s.startswith("[web]")]
                summary = self._send([{"role": "user", "content": (
                    f"You just answered this query using web search: {query}\n\n"
                    f"Sources used: {web_sources}\n\n"
                    "Write a clean, factual summary of the information you found from these sources. "
                    "Write it as plain prose suitable for storing in a knowledge base. "
                    "Do not include any encrypted content, URLs, or formatting artifacts."
                )}])
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                    tmp.write(summary)
                    tmp_path = tmp.name
                try:
                    self._ingestion_agent.ingest(tmp_path)
                finally:
                    os.unlink(tmp_path)

        if len(self._chat_history) % 6 == 0:
            self._add_context(self._chat_history[-10:])

        return RAGResponse(answer=answer, sources=sources)

    def _search_web(self, query: str) -> tuple[str, list[str]]:
        response = self._client.messages.create(
            model=self._model,
            system=self._system_prompt,
            messages=[{"role": "user", "content": query}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            max_tokens=4096,
        )

        context_parts: list[str] = []
        sources: list[str] = []
        recent_history = self._chat_history[-6:]
        formatted_history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history
        )

        for block in response.content:
            if block.type == "web_search_tool_result":
                for result in block.content:
                    if result.type == "web_search_result":
                        relevance_prompt = (
                            f"Recent conversation:\n{formatted_history}\n\n"
                            f"Current query: {query}\n\n"
                            f"Web result URL: {result.url}\n"
                            f"Web result title: {result.title}\n\n"
                            "Is this web result relevant to the query and conversation context? Reply with one word: 'yes' or 'no'."
                        )
                        is_relevant = self._send([{"role": "user", "content": relevance_prompt}]).strip().lower()
                        if "yes" in is_relevant[:10]:
                            context_parts.append(f"[WEB] {result.url}\n{result.encrypted_content}")
                            sources.append(f"[web] {result.url}")

        return "\n\n---\n\n".join(context_parts), sources
        
