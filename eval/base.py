import anthropic
from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeScore:
    score: int        # 1-5
    reasoning: str


class Judge:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model_name
        self._system_prompt = system_prompt

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    def _send(self, messages: list, max_tokens: int = 1024, **kwargs) -> str:
        response = self._client.messages.create(
            model=self._model,
            system=self._system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text

    def score(self, **kwargs) -> JudgeScore:
        raise NotImplementedError
