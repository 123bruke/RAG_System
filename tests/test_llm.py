"""LLM provider wrappers, tested with fake SDK clients (no network, no keys)."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from app.exceptions import LLMAuthError, LLMError
from app.generation.llm import GeminiLLM, OpenAILLM


def _openai_status_error(cls, status):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


def _fake_openai(behaviour):
    create = behaviour if callable(behaviour) else (lambda **kw: behaviour)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _fake_gemini(behaviour):
    generate = behaviour if callable(behaviour) else (lambda **kw: behaviour)
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate))


def _openai_llm(client):
    llm = OpenAILLM("sk-test", "gpt-test")
    llm._client = client
    return llm


def _gemini_llm(client):
    llm = GeminiLLM("AIza-test", "gemini-test")
    llm._client = client
    return llm


def test_openai_returns_text_and_sends_both_prompts():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  Hello [1] "))])

    assert _openai_llm(_fake_openai(create)).generate("SYS", "USER") == "Hello [1]"
    assert seen["model"] == "gpt-test"
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


def test_openai_invalid_key_maps_to_auth_error():
    def create(**kwargs):
        raise _openai_status_error(openai.AuthenticationError, 401)

    with pytest.raises(LLMAuthError):
        _openai_llm(_fake_openai(create)).generate("s", "u")


def test_openai_rate_limit_and_empty_answer_map_to_llm_error():
    def limited(**kwargs):
        raise _openai_status_error(openai.RateLimitError, 429)

    with pytest.raises(LLMError, match="rate limit"):
        _openai_llm(_fake_openai(limited)).generate("s", "u")

    empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    with pytest.raises(LLMError, match="empty"):
        _openai_llm(_fake_openai(empty)).generate("s", "u")


def test_gemini_returns_text():
    llm = _gemini_llm(_fake_gemini(SimpleNamespace(text=" Answer [1] ")))
    assert llm.generate("SYS", "USER") == "Answer [1]"


@pytest.mark.parametrize(
    "code,payload,expected",
    [
        (400, {"error": {"message": "API key not valid. Please pass a valid API key."}}, LLMAuthError),
        (403, {"error": {"message": "forbidden"}}, LLMAuthError),
        (429, {"error": {"message": "quota"}}, LLMError),
        (500, {"error": {"message": "internal"}}, LLMError),
    ],
)
def test_gemini_error_mapping(code, payload, expected):
    def generate(**kwargs):
        raise genai_errors.APIError(code, payload)

    with pytest.raises(expected) as info:
        _gemini_llm(_fake_gemini(generate)).generate("s", "u")
    assert type(info.value) is expected  # LLMAuthError only for key problems


def test_gemini_blocked_or_empty_response_is_an_error():
    with pytest.raises(LLMError, match="empty"):
        _gemini_llm(_fake_gemini(SimpleNamespace(text=None))).generate("s", "u")


def test_real_sdk_clients_can_be_constructed_offline():
    """Guards against SDK API drift: building the client makes no network call."""
    assert OpenAILLM("sk-test", "m").client is not None
    assert GeminiLLM("AIza-test", "m").client is not None
