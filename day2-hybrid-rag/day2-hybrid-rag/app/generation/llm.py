"""LLM abstraction.

The pipeline only knows the tiny :class:`BaseLLM` interface (``generate``).
Concrete providers hide their SDK details behind it, so switching provider is a
one-line change in ``.env`` (``LLM_PROVIDER=gemini`` or ``openai``) and unit
tests can plug in a fake. SDKs are imported lazily, so you only need the
package of the provider you actually use.

API keys come from settings/.env - never from source code.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.config import Settings
from app.exceptions import ConfigurationError, LLMAuthError, LLMError

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Minimal interface: system prompt + user prompt in, answer text out."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def _require_key(api_key: str, env_name: str, provider: str) -> str:
    if not api_key:
        raise LLMAuthError(
            f"No {provider} API key configured. Set {env_name} in your .env file "
            "(copy .env.example to .env)."
        )
    return api_key


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout: float = 60.0) -> None:
        self.api_key = _require_key(api_key, "OPENAI_API_KEY", "OpenAI")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigurationError("The 'openai' package is not installed.") from exc
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LLMAuthError("OpenAI rejected the API key (check OPENAI_API_KEY).") from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenAI rate limit or quota exceeded. Try again later.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Could not reach OpenAI: {exc}") from exc
        except openai.APIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("OpenAI returned an empty answer.")
        return text


class GeminiLLM(BaseLLM):
    """Google Gemini through the official ``google-genai`` SDK."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout: float = 60.0) -> None:
        self.api_key = _require_key(api_key, "GOOGLE_API_KEY", "Gemini")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise ConfigurationError("The 'google-genai' package is not installed.") from exc
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),  # milliseconds
            )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import errors, types

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                ),
            )
        except errors.APIError as exc:
            message = str(getattr(exc, "message", "") or exc)
            if getattr(exc, "code", None) in (401, 403) or "api key" in message.lower():
                raise LLMAuthError("Gemini rejected the API key (check GOOGLE_API_KEY).") from exc
            if getattr(exc, "code", None) == 429:
                raise LLMError("Gemini rate limit or quota exceeded. Try again later.") from exc
            raise LLMError(f"Gemini request failed: {message}") from exc
        except Exception as exc:  # network errors, timeouts
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty answer (it may have been blocked by safety filters).")
        return text


def create_llm(settings: Settings) -> BaseLLM:
    """Build the LLM provider selected in settings."""
    model = settings.resolved_llm_model
    logger.info("Using LLM provider=%s model=%s", settings.llm_provider, model)
    if settings.llm_provider == "gemini":
        return GeminiLLM(
            settings.active_api_key, model, settings.llm_temperature, settings.llm_timeout_seconds
        )
    if settings.llm_provider == "openai":
        return OpenAILLM(
            settings.active_api_key, model, settings.llm_temperature, settings.llm_timeout_seconds
        )
    raise ConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")
