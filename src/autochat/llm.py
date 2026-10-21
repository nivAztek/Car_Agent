"""Foundry chat + embedding client factories (langchain-azure-ai, OpenAI v1 endpoint)."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any, Literal

import openai
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from autochat.config import get_settings

# Structured output (json_schema) returns ParsedChatCompletion whose `parsed` field is typed None in the SDK model;
# serializing it for tracing emits a harmless pydantic warning on every call. Silence exactly that one.
warnings.filterwarnings(
    "ignore", message=r"(?s)Pydantic serializer warnings:.*field_name='parsed'", category=UserWarning
)

ModelRole = Literal["chat", "fast", "extract"]

# Transient Foundry errors worth retrying (429, timeouts, connection resets, 5xx).
RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
RETRY_KW: dict[str, Any] = {
    "retry": retry_if_exception_type(RETRYABLE),
    "stop": stop_after_attempt(5),
    "wait": wait_exponential_jitter(multiplier=2, max=30),
    "reraise": True,
}


def _credential() -> Any:
    s = get_settings()
    if s.foundry_api_key and s.foundry_api_key.get_secret_value():
        return s.foundry_api_key.get_secret_value()
    from azure.identity import DefaultAzureCredential  # Entra ID fallback (az login)

    return DefaultAzureCredential()


def _endpoint() -> str:
    s = get_settings()
    if not s.foundry_endpoint:
        raise RuntimeError("FOUNDRY_ENDPOINT is not set (see .env.example)")
    return s.foundry_endpoint.rstrip("/")


@lru_cache
def chat_llm(role: ModelRole = "chat", temperature: float | None = None) -> AzureAIOpenAIApiChatModel:
    """Chat model for a role. `temperature` is passed only when given (reasoning models may reject it)."""
    s = get_settings()
    model = {"chat": s.foundry_chat_deployment, "fast": s.fast_deployment, "extract": s.extract_deployment}[role]
    if not model:
        raise RuntimeError(f"No deployment configured for role {role!r}")
    kwargs: dict[str, Any] = {} if temperature is None else {"temperature": temperature}
    return AzureAIOpenAIApiChatModel(endpoint=_endpoint(), credential=_credential(), model=model, **kwargs)


@lru_cache
def embeddings() -> AzureAIOpenAIApiEmbeddingsModel:
    s = get_settings()
    return AzureAIOpenAIApiEmbeddingsModel(
        endpoint=_endpoint(),
        credential=_credential(),
        model=s.foundry_embed_deployment,
        dimensions=s.embed_dimensions,
    )
