"""Configuration loader for ACE.

Loads from configs/default.toml and overrides with environment variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


@dataclass
class DatabaseConfig:
    url: str


@dataclass
class EmbeddingsConfig:
    model: str


@dataclass
class RetrievalConfig:
    top_k: int
    lexical_weight: float
    max_bullets: int


@dataclass
class RefineConfig:
    threshold: float
    minhash_threshold: float


@dataclass
class TrainingConfig:
    gate_on_regression: bool
    max_regression_delta: float
    held_out_path: str
    regression_metrics: list[str]


@dataclass
class LoggingConfig:
    level: str
    format: str


@dataclass
class MCPConfig:
    transport: str
    port: int


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass
class ACEConfig:
    database: DatabaseConfig
    embeddings: EmbeddingsConfig
    retrieval: RetrievalConfig
    refine: RefineConfig
    training: TrainingConfig
    logging: LoggingConfig
    mcp: MCPConfig
    llm: LLMConfig


def _validate_config(config: ACEConfig) -> None:
    """Validate configuration values.

    Args:
        config: ACEConfig to validate

    Raises:
        ValueError: If validation fails
    """
    # Validate retrieval values
    if config.retrieval.top_k < 1:
        raise ValueError(f"retrieval.top_k must be >= 1, got {config.retrieval.top_k}")
    if not 0.0 <= config.retrieval.lexical_weight <= 1.0:
        val = config.retrieval.lexical_weight
        raise ValueError(f"retrieval.lexical_weight must be in [0.0, 1.0], got {val}")

    # Validate refine thresholds
    if not 0.0 <= config.refine.threshold <= 1.0:
        raise ValueError(f"refine.threshold must be in [0.0, 1.0], got {config.refine.threshold}")
    if not 0.0 <= config.refine.minhash_threshold <= 1.0:
        val = config.refine.minhash_threshold
        raise ValueError(f"refine.minhash_threshold must be in [0.0, 1.0], got {val}")

    # Validate logging level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if config.logging.level.upper() not in valid_levels:
        raise ValueError(f"logging.level must be one of {valid_levels}, got {config.logging.level}")

    # Validate MCP transport
    valid_transports = {"stdio", "http", "sse"}
    if config.mcp.transport not in valid_transports:
        msg = f"mcp.transport must be one of {valid_transports}, got {config.mcp.transport}"
        raise ValueError(msg)
    if config.mcp.port < 1 or config.mcp.port > 65535:
        raise ValueError(f"mcp.port must be in [1, 65535], got {config.mcp.port}")

    # Validate LLM values
    if config.llm.temperature < 0.0 or config.llm.temperature > 2.0:
        raise ValueError(f"llm.temperature must be in [0.0, 2.0], got {config.llm.temperature}")
    if config.llm.max_tokens < 1:
        raise ValueError(f"llm.max_tokens must be >= 1, got {config.llm.max_tokens}")


def load_config(config_path: Path | None = None) -> ACEConfig:
    """Load configuration from TOML file and override with env vars.

    Args:
        config_path: Path to TOML config file. Defaults to configs/default.toml

    Returns:
        ACEConfig instance with merged configuration

    Raises:
        ValueError: If configuration validation fails
    """
    if config_path is None:
        # Default to configs/default.toml relative to project root
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.toml"

    # Load TOML file
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)

    # Override with environment variables (ACE_ prefix)
    database_url = os.getenv("ACE_DB_URL", config_dict["database"]["url"])
    embeddings_model = os.getenv("ACE_EMBEDDINGS", config_dict["embeddings"]["model"])
    retrieval_topk = int(os.getenv("ACE_RETRIEVAL_TOPK", config_dict["retrieval"]["top_k"]))
    retrieval_lexical = float(
        os.getenv("ACE_RETRIEVAL_LEXICAL_WEIGHT", config_dict["retrieval"]["lexical_weight"])
    )
    retrieval_max_bullets = int(
        os.getenv("ACE_RETRIEVAL_MAX_BULLETS", config_dict["retrieval"].get("max_bullets", 2000))
    )
    refine_threshold = float(os.getenv("ACE_REFINE_THRESHOLD", config_dict["refine"]["threshold"]))
    refine_minhash = float(
        os.getenv("ACE_REFINE_MINHASH_THRESHOLD", config_dict["refine"]["minhash_threshold"])
    )

    training_dict = config_dict.get("training", {})
    training_gate = os.getenv(
        "ACE_TRAINING_GATE_ON_REGRESSION",
        str(training_dict.get("gate_on_regression", False))
    ).lower() in ("true", "1", "yes")
    training_max_delta = float(os.getenv(
        "ACE_TRAINING_MAX_REGRESSION_DELTA",
        training_dict.get("max_regression_delta", 0.05)
    ))
    training_held_out = os.getenv(
        "ACE_TRAINING_HELD_OUT_PATH",
        training_dict.get("held_out_path", "")
    )
    training_metrics_str = os.getenv(
        "ACE_TRAINING_REGRESSION_METRICS",
        training_dict.get("regression_metrics", "mrr,recall")
    )
    training_metrics = [m.strip() for m in training_metrics_str.split(",") if m.strip()]

    log_level = os.getenv("ACE_LOG_LEVEL", config_dict["logging"]["level"])
    log_format = os.getenv("ACE_LOG_FORMAT", config_dict["logging"]["format"])
    mcp_transport = os.getenv("MCP_TRANSPORT", config_dict["mcp"]["transport"])
    mcp_port = int(os.getenv("MCP_PORT", config_dict["mcp"]["port"]))
    llm_provider = os.getenv("ACE_LLM_PROVIDER", config_dict["llm"]["provider"])
    llm_model = os.getenv("ACE_LLM_MODEL", config_dict["llm"]["model"])
    llm_temp = float(os.getenv("ACE_LLM_TEMPERATURE", config_dict["llm"]["temperature"]))
    llm_max_tokens = int(os.getenv("ACE_LLM_MAX_TOKENS", config_dict["llm"]["max_tokens"]))

    config = ACEConfig(
        database=DatabaseConfig(url=database_url),
        embeddings=EmbeddingsConfig(model=embeddings_model),
        retrieval=RetrievalConfig(
            top_k=retrieval_topk,
            lexical_weight=retrieval_lexical,
            max_bullets=retrieval_max_bullets,
        ),
        refine=RefineConfig(threshold=refine_threshold, minhash_threshold=refine_minhash),
        training=TrainingConfig(
            gate_on_regression=training_gate,
            max_regression_delta=training_max_delta,
            held_out_path=training_held_out,
            regression_metrics=training_metrics,
        ),
        logging=LoggingConfig(level=log_level, format=log_format),
        mcp=MCPConfig(transport=mcp_transport, port=mcp_port),
        llm=LLMConfig(
            provider=llm_provider,
            model=llm_model,
            temperature=llm_temp,
            max_tokens=llm_max_tokens
        )
    )

    # Validate before returning
    _validate_config(config)

    return config


# Global config instance
_config: ACEConfig | None = None


def get_config() -> ACEConfig:
    """Get the global config instance, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
