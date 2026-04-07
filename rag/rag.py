from agent.base import Agent
from typing import Optional
from dataclasses import dataclass


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]


class RAGAgent(Agent):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        knowledge_dir: str,
        system_prompt: str,
        chat_history: Optional[list] = None,
    ):
        super().__init__(api_key, model_name, system_prompt, chat_history)
        self._knowledge_dir = knowledge_dir

    def add_file(self, file_path: str) -> None:
        raise NotImplementedError

    def add_dir(self, dir_path: str) -> None:
        raise NotImplementedError

    def respond(self, query: str) -> RAGResponse:
        raise NotImplementedError
