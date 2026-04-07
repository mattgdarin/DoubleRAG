import anthropic
from typing import Optional


class Agent:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        chat_history: Optional[list] = None,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model_name
        self._system_prompt = system_prompt
        self._chat_history: list = chat_history or []

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    def _send(self, messages: list, max_tokens: int = 4096, **kwargs) -> str:
        response = self._client.messages.create(
            model=self._model,
            system=self._system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text

    def _stream(self, messages: list, max_tokens: int = 4096, **kwargs) -> str:
        full_text = ""
        with self._client.messages.stream(
            model=self._model,
            system=self._system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        ) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                full_text += chunk
        print()
        return full_text
