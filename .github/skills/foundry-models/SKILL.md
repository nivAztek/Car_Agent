---
name: foundry-models
description: "Connect autochat to Microsoft Foundry chat and embedding deployments with langchain-azure-ai: client factories, structured output, streaming, embeddings dimensions, connectivity smoke test and error triage. Use when writing src/autochat/llm.py, when LLM/embedding calls fail, or when changing models."
---

# foundry-models

## Smoke test (do this first, after filling `.env`)

```powershell
.venv\Scripts\python.exe .github/skills/foundry-models/scripts/foundry_smoke.py
```
Checks chat, Pydantic structured output, embeddings (length = `EMBED_DIMENSIONS`, plus Hebrew↔English similarity), and token streaming. It never prints the key.

## Configuration (`.env`)

| Var | Meaning |
|---|---|
| `FOUNDRY_ENDPOINT` | `https://<resource>.services.ai.azure.com/openai/v1` (or `...openai.azure.com/openai/v1`). Used **verbatim** as `base_url`, so it must end with `/openai/v1`. |
| `FOUNDRY_API_KEY` | Key. If empty, use `DefaultAzureCredential()` (needs `az login` + the *Cognitive Services OpenAI User* role). |
| `FOUNDRY_CHAT_DEPLOYMENT` | Generation model, for answers. |
| `FOUNDRY_FAST_DEPLOYMENT` | Cheaper model for `analyze_turn`, grading, translation, and extraction. Empty means use CHAT. |
| `FOUNDRY_EMBED_DEPLOYMENT` | `text-embedding-3-large`, called with `dimensions=EMBED_DIMENSIONS` (1536). |

⚠️ Don't set `AZURE_AI_PROJECT_ENDPOINT`. `langchain-azure-ai` gives it precedence over the `endpoint` parameter.

## Client factory (verified signatures, langchain-azure-ai 1.2.x)

```python
from functools import lru_cache
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel


def _cred(s):  # str key or Entra credential
    if s.foundry_api_key:
        return s.foundry_api_key.get_secret_value()
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


@lru_cache
def chat_llm(fast: bool = False, temperature: float | None = None) -> AzureAIOpenAIApiChatModel:
    s = get_settings()
    kw = {} if temperature is None else {"temperature": temperature}
    return AzureAIOpenAIApiChatModel(
        endpoint=s.foundry_endpoint,
        credential=_cred(s),
        model=(s.foundry_fast_deployment or s.foundry_chat_deployment) if fast else s.foundry_chat_deployment,
        **kw,
    )


@lru_cache
def embeddings() -> AzureAIOpenAIApiEmbeddingsModel:
    s = get_settings()
    return AzureAIOpenAIApiEmbeddingsModel(
        endpoint=s.foundry_endpoint,
        credential=_cred(s),
        model=s.foundry_embed_deployment,
        dimensions=s.embed_dimensions,
    )
```
Available fields: `endpoint | project_endpoint` (mutually exclusive), `credential`, `model`, `temperature`, `max_tokens`, `reasoning_effort`, `use_responses_api`, `api_version`. Embeddings: `model`, `dimensions`.

**Reasoning models** (gpt-5.x family) may reject `temperature != 1`. Pass `temperature` only when the deployment accepts it. Otherwise use `reasoning_effort="low"` for the fast path. If the smoke test fails with a 400 on temperature, drop the parameter.

## Usage patterns
- Structured output: `chat_llm(fast=True).with_structured_output(TurnAnalysis)` then `await .ainvoke(messages)`. Keep schemas flat, add a `Field(description=...)` for every field, and use `Literal[...]` for enums.
- Streaming for the answer node: call `chat_llm().astream(messages)` inside the node. LangGraph `stream_mode="messages"` surfaces the tokens to FastAPI automatically.
- Embeddings: `await embeddings().aembed_documents(texts)` in batches of 16 (ingestion); `await embeddings().aembed_query(q)` at runtime. Vectors from `text-embedding-3-*` are already unit-normalized.
- Wrap ingestion calls with `tenacity` retry on `openai.RateLimitError` / `APITimeoutError` / `APIConnectionError`.

## Error triage
| Symptom | Fix |
|---|---|
| 404 `DeploymentNotFound` | The deployment name is wrong (it's not the model name), or the endpoint belongs to another resource. |
| 404 on the path | The endpoint is missing `/openai/v1`. |
| 401 | Wrong key, or (Entra) missing the role. |
| 400 `temperature` / `max_tokens` unsupported | Reasoning model: remove `temperature`; use `max_completion_tokens` via `model_kwargs` if needed. |
| 429 | Back off (tenacity) and lower ingestion concurrency to 2. |
| Embedding length ≠ container dims | Make `EMBED_DIMENSIONS` match, then run `cosmos_admin.py reset` and re-embed. |
