# LLM Smoke Test Command

## Overview

The `ace smoke-test-model` command performs a simple connectivity test to verify that your configured LLM provider is working correctly. This is useful for validating your API keys and network connectivity before running more complex operations.

## Usage

### Basic Usage

```bash
ace smoke-test-model
```

This will:
1. Load your LLM configuration from `configs/default.toml` and environment variables
2. Create an LLM client for the configured provider
3. Send a simple test message
4. Display the response and success status

### JSON Output

```bash
ace smoke-test-model --json
```

Returns structured JSON output suitable for automation and scripting.

## Configuration

The command uses your existing LLM configuration. Configure via environment variables:

### For OpenRouter

```bash
export OPENROUTER_API_KEY="your-api-key-here"
export ACE_LLM_PROVIDER="openrouter"
export ACE_LLM_MODEL="openai/gpt-4o-mini"
```

### Configuration File

Alternatively, edit `configs/default.toml`:

```toml
[llm]
provider = "openrouter"
model = "openai/gpt-4o-mini"
temperature = 0.0
max_tokens = 2000
```

## Example Output

### Successful Test

```
Testing LLM provider: openrouter
Model: openai/gpt-4o-mini
Temperature: 0.0
Max tokens: 2000

✓ LLM client created successfully
Sending test request to LLM...
✓ LLM request successful

Response:
------------------------------------------------------------
Hello from ACE! I can read this message.
------------------------------------------------------------

✓ Smoke test PASSED
  Provider 'openrouter' is working correctly
```

### Failed Test

```
Testing LLM provider: openrouter
Model: openai/gpt-4o-mini
Temperature: 0.0
Max tokens: 2000

✗ Configuration error: OpenRouter API key must be provided via api_key
parameter or OPENROUTER_API_KEY environment variable

Troubleshooting tips:
  1. Check your API key environment variable
     - For OpenRouter: OPENROUTER_API_KEY
  2. Verify your network connection
  3. Ensure the model name is correct
     - Current model: openai/gpt-4o-mini
```

## Supported Providers

Currently supported providers:
- **openrouter**: Access to multiple model providers through OpenRouter
- **mock**: Mock provider for testing (doesn't require API keys)

## Common Issues

### Missing API Key

**Error**: `OpenRouter API key must be provided`

**Solution**: Set the `OPENROUTER_API_KEY` environment variable:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Network Connection Errors

**Error**: `LLM request failed: Network error`

**Solution**:
- Check your internet connection
- Verify firewall settings allow HTTPS connections
- Try with a different network if behind a proxy

### Invalid Model Name

**Error**: `Model not found` or `Invalid model`

**Solution**: Verify the model name is correct. For OpenRouter, use the format:
- `openai/gpt-4o-mini`
- `anthropic/claude-3-opus`
- See https://openrouter.ai/docs for available models

## Integration in Workflows

Use this command before running longer operations to validate configuration:

```bash
#!/bin/bash
# Validate LLM setup before training
if ace smoke-test-model --json | grep -q '"status": "success"'; then
    echo "LLM provider validated, starting training..."
    ace train --data training_data.jsonl --epochs 3
else
    echo "LLM provider test failed. Fix configuration before proceeding."
    exit 1
fi
```

## Related Commands

- `ace version` - Show ACE version
- `ace stats` - Show playbook statistics
- `ace pipeline <query>` - Run full ACE pipeline (requires working LLM)
