# Custom LLM Provider Example

ACE supports two environment-configured providers today: `mock` and `openrouter`.

If you want to use a different provider or an in-house gateway, implement the `LLMClient` interface and inject that client into the ACE component you are calling. This keeps the extension local and avoids forking the reflector or pipeline.

## Run the executable example

From the repo root:

```bash
.venv/bin/python scripts/custom_llm_provider_example.py
```

Expected output:

```text
Custom provider reflection summary
Error: Pytest failed after a missing retry policy
Insight: Custom providers only need to implement the LLMClient interface.
Bullet tags: 1
Candidate bullets: 1
```

## Minimal implementation

[`scripts/custom_llm_provider_example.py`](../scripts/custom_llm_provider_example.py) shows the complete example. The important seam is:

```python
from ace.llm import CompletionResponse, LLMClient


class MyProviderClient(LLMClient):
    def complete(self, messages, **kwargs) -> CompletionResponse:
        payload = call_my_provider(messages)
        return CompletionResponse(text=payload["text"])
```

Then inject it:

```python
from ace.pipeline import Pipeline
from ace.reflector import Reflector

client = MyProviderClient()

reflector = Reflector(llm_client=client)
pipeline = Pipeline(store=my_store, llm_client=client)
```

## When to use this pattern

- You need a provider ACE does not natively construct from config.
- You want custom auth, routing, or observability around model calls.
- You want to test reflection or pipeline behavior with deterministic responses.

## Current limitation

The `ACE_LLM_PROVIDER` config knob does not dynamically load arbitrary providers. For non-built-in providers, use injected `LLMClient` instances as shown above.
