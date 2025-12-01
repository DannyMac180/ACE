"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest

from ace.core.config import load_config


def test_load_default_config():
    """Test loading default config file."""
    config = load_config()

    assert config.database.url == "sqlite:///ace.db"
    assert config.embeddings.model == "bge-small"
    assert config.retrieval.top_k == 24
    assert config.refine.threshold == 0.90
    assert config.logging.level == "INFO"


def test_env_override():
    """Test environment variable overrides."""
    os.environ["ACE_DB_URL"] = "postgres://test"
    os.environ["ACE_RETRIEVAL_TOPK"] = "50"

    try:
        config = load_config()
        assert config.database.url == "postgres://test"
        assert config.retrieval.top_k == 50
    finally:
        del os.environ["ACE_DB_URL"]
        del os.environ["ACE_RETRIEVAL_TOPK"]


def test_custom_config_path():
    """Test loading from custom config path."""
    # Clear any env overrides for this test
    old_llm_provider = os.environ.pop("ACE_LLM_PROVIDER", None)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
[database]
url = "sqlite:///custom.db"

[embeddings]
model = "custom-model"

[retrieval]
top_k = 10
lexical_weight = 0.3

[refine]
threshold = 0.85
minhash_threshold = 0.80

[logging]
level = "DEBUG"
format = "text"

[mcp]
transport = "http"
port = 9000

[llm]
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
temperature = 0.1
max_tokens = 4000
""")
        temp_path = Path(f.name)

    try:
        config = load_config(temp_path)
        assert config.database.url == "sqlite:///custom.db"
        assert config.embeddings.model == "custom-model"
        assert config.retrieval.top_k == 10
        assert config.refine.threshold == 0.85
        assert config.llm.provider == "anthropic"
    finally:
        temp_path.unlink()
        # Restore the env var if it was set
        if old_llm_provider is not None:
            os.environ["ACE_LLM_PROVIDER"] = old_llm_provider


def test_config_types():
    """Test that config values have correct types."""
    config = load_config()

    assert isinstance(config.retrieval.top_k, int)
    assert isinstance(config.retrieval.lexical_weight, float)
    assert isinstance(config.refine.threshold, float)
    assert isinstance(config.mcp.port, int)
    assert isinstance(config.llm.temperature, float)
    assert isinstance(config.llm.max_tokens, int)


def test_validation_retrieval_top_k():
    """Test validation of retrieval.top_k."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
[database]
url = "sqlite:///test.db"
[embeddings]
model = "test"
[retrieval]
top_k = 0
lexical_weight = 0.5
[refine]
threshold = 0.9
minhash_threshold = 0.85
[logging]
level = "INFO"
format = "json"
[mcp]
transport = "stdio"
port = 8000
[llm]
provider = "openai"
model = "gpt-4"
temperature = 0.0
max_tokens = 1000
""")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="retrieval.top_k must be >= 1"):
            load_config(temp_path)
    finally:
        temp_path.unlink()


def test_validation_lexical_weight():
    """Test validation of retrieval.lexical_weight."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
[database]
url = "sqlite:///test.db"
[embeddings]
model = "test"
[retrieval]
top_k = 10
lexical_weight = 1.5
[refine]
threshold = 0.9
minhash_threshold = 0.85
[logging]
level = "INFO"
format = "json"
[mcp]
transport = "stdio"
port = 8000
[llm]
provider = "openai"
model = "gpt-4"
temperature = 0.0
max_tokens = 1000
""")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="lexical_weight must be in"):
            load_config(temp_path)
    finally:
        temp_path.unlink()


def test_validation_invalid_log_level():
    """Test validation of invalid logging level."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
[database]
url = "sqlite:///test.db"
[embeddings]
model = "test"
[retrieval]
top_k = 10
lexical_weight = 0.5
[refine]
threshold = 0.9
minhash_threshold = 0.85
[logging]
level = "INVALID"
format = "json"
[mcp]
transport = "stdio"
port = 8000
[llm]
provider = "openai"
model = "gpt-4"
temperature = 0.0
max_tokens = 1000
""")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="logging.level must be one of"):
            load_config(temp_path)
    finally:
        temp_path.unlink()
