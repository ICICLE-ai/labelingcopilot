"""
Configuration management for the data augmentation tool.
Loads settings from environment variables with sensible defaults.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class AzureConfig:
    """Provider configuration for Azure OpenAI or the standard OpenAI API."""

    provider: str
    vision_api_key: str
    image_edit_api_key: str

    # Azure-specific fields
    vision_endpoint: str = ""
    image_edit_endpoint: str = ""
    vision_deployment: str = "gpt-5.4-nano"
    vision_api_version: str = "2024-12-01-preview"
    image_edit_deployment: str = "gpt-image-1.5"
    image_edit_api_version: str = "2025-04-01-preview"

    # Standard OpenAI-specific fields
    openai_base_url: str = "https://api.openai.com/v1"
    vision_model: str = "gpt-5.4-nano"
    image_edit_model: str = "gpt-image-1.5"

    @property
    def is_azure(self) -> bool:
        return self.provider == "azure"

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Load provider configuration from environment variables."""
        provider = (os.getenv("MODEL_API_PROVIDER") or "").strip().lower()

        azure_fields = {
            "AZURE_VISION_ENDPOINT": os.getenv("AZURE_VISION_ENDPOINT"),
            "AZURE_VISION_API_KEY": os.getenv("AZURE_VISION_API_KEY"),
            "AZURE_IMAGE_EDIT_ENDPOINT": os.getenv("AZURE_IMAGE_EDIT_ENDPOINT"),
            "AZURE_IMAGE_EDIT_API_KEY": os.getenv("AZURE_IMAGE_EDIT_API_KEY"),
        }
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_fields = {
            "OPENAI_API_KEY": openai_key,
            "OPENAI_VISION_API_KEY": os.getenv("OPENAI_VISION_API_KEY"),
            "OPENAI_IMAGE_EDIT_API_KEY": os.getenv("OPENAI_IMAGE_EDIT_API_KEY"),
        }

        has_azure = all(azure_fields.values())
        has_openai = any(openai_fields.values())

        if not provider:
            if has_azure and has_openai:
                raise ValueError(
                    "Both Azure and OpenAI credentials were detected. Set MODEL_API_PROVIDER=azure or MODEL_API_PROVIDER=openai."
                )
            if has_azure:
                provider = "azure"
            elif has_openai:
                provider = "openai"
            else:
                raise ValueError(
                    "Missing model API configuration. Provide either Azure OpenAI or OpenAI credentials.\n"
                    "Azure variables:\n"
                    "  AZURE_VISION_ENDPOINT\n"
                    "  AZURE_VISION_API_KEY\n"
                    "  AZURE_IMAGE_EDIT_ENDPOINT\n"
                    "  AZURE_IMAGE_EDIT_API_KEY\n"
                    "OpenAI variables:\n"
                    "  OPENAI_API_KEY\n"
                    "Optional OpenAI overrides:\n"
                    "  OPENAI_VISION_API_KEY\n"
                    "  OPENAI_IMAGE_EDIT_API_KEY\n"
                    "  OPENAI_VISION_MODEL\n"
                    "  OPENAI_IMAGE_EDIT_MODEL"
                )

        if provider == "azure":
            if not has_azure:
                raise ValueError(
                    "Missing required Azure configuration. Please set:\n"
                    "  AZURE_VISION_ENDPOINT\n"
                    "  AZURE_VISION_API_KEY\n"
                    "  AZURE_IMAGE_EDIT_ENDPOINT\n"
                    "  AZURE_IMAGE_EDIT_API_KEY"
                )

            return cls(
                provider="azure",
                vision_endpoint=azure_fields["AZURE_VISION_ENDPOINT"] or "",
                vision_api_key=azure_fields["AZURE_VISION_API_KEY"] or "",
                vision_deployment=os.getenv("AZURE_VISION_DEPLOYMENT", "gpt-5.4-nano"),
                vision_api_version=os.getenv("AZURE_VISION_API_VERSION", "2024-12-01-preview"),
                image_edit_endpoint=azure_fields["AZURE_IMAGE_EDIT_ENDPOINT"] or "",
                image_edit_api_key=azure_fields["AZURE_IMAGE_EDIT_API_KEY"] or "",
                image_edit_deployment=os.getenv("AZURE_IMAGE_EDIT_DEPLOYMENT", "gpt-image-1.5"),
                image_edit_api_version=os.getenv("AZURE_IMAGE_EDIT_API_VERSION", "2025-04-01-preview"),
            )

        if provider == "openai":
            vision_api_key = os.getenv("OPENAI_VISION_API_KEY", openai_key or "")
            image_edit_api_key = os.getenv("OPENAI_IMAGE_EDIT_API_KEY", openai_key or "")

            if not vision_api_key or not image_edit_api_key:
                raise ValueError(
                    "Missing required OpenAI configuration. Please set:\n"
                    "  OPENAI_API_KEY\n"
                    "Or provide both of:\n"
                    "  OPENAI_VISION_API_KEY\n"
                    "  OPENAI_IMAGE_EDIT_API_KEY"
                )

            return cls(
                provider="openai",
                vision_api_key=vision_api_key,
                image_edit_api_key=image_edit_api_key,
                openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                vision_model=os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4-nano")),
                image_edit_model=os.getenv("OPENAI_IMAGE_EDIT_MODEL", "gpt-image-1.5"),
            )

        raise ValueError("Unsupported MODEL_API_PROVIDER. Use 'azure' or 'openai'.")


@dataclass
class ProcessingConfig:
    """Image processing configuration."""

    batch_size: int = 10
    max_concurrent: int = 3
    rate_limit_delay: float = 3.0
    requests_per_minute: int = 20
    max_retries: int = 3
    min_file_size: int = 1024

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        """Load configuration from environment variables."""
        # Calculate rate limit delay from requests per minute
        requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", "20"))
        rate_limit_delay = 60.0 / requests_per_minute  # Convert to delay in seconds

        return cls(
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_concurrent=int(os.getenv("MAX_CONCURRENT", "1")),  # Use 1 to respect rate limit
            rate_limit_delay=rate_limit_delay,
            requests_per_minute=requests_per_minute,
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            min_file_size=int(os.getenv("MIN_FILE_SIZE", "1024")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
        )


@dataclass
class ForteConfig:
    """Forte OOD detection configuration."""

    enabled: bool = False
    method: str = "gmm"  # gmm, kde, or ocsvm
    k_neighbors: int = 5
    threshold: float = 0.3
    embedding_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ForteConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("FORTE_ENABLED", "false").lower() == "true",
            method=os.getenv("FORTE_METHOD", "gmm"),
            k_neighbors=int(os.getenv("FORTE_K_NEIGHBORS", "5")),
            threshold=float(os.getenv("FORTE_THRESHOLD", "0.3")),
            embedding_dir=os.getenv("FORTE_EMBEDDING_DIR"),
        )


@dataclass
class EvaluationConfig:
    """Quality evaluation configuration."""

    # Metrics to compute
    compute_prdc: bool = True
    compute_fd: bool = True
    compute_fls: bool = False  # Requires baseline
    compute_inception_score: bool = True
    compute_authpct: bool = True
    compute_ct_score: bool = False  # Expensive
    compute_mmd: bool = False  # Expensive
    compute_vendi: bool = True
    compute_sw: bool = True

    # PRDC parameters
    prdc_k_neighbors: int = 5

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Load configuration from environment variables."""
        return cls(
            compute_prdc=os.getenv("EVAL_PRDC", "true").lower() == "true",
            compute_fd=os.getenv("EVAL_FD", "true").lower() == "true",
            compute_fls=os.getenv("EVAL_FLS", "false").lower() == "true",
            compute_inception_score=os.getenv("EVAL_INCEPTION", "true").lower() == "true",
            compute_authpct=os.getenv("EVAL_AUTHPCT", "true").lower() == "true",
            compute_ct_score=os.getenv("EVAL_CT", "false").lower() == "true",
            compute_mmd=os.getenv("EVAL_MMD", "false").lower() == "true",
            compute_vendi=os.getenv("EVAL_VENDI", "true").lower() == "true",
            compute_sw=os.getenv("EVAL_SW", "true").lower() == "true",
            prdc_k_neighbors=int(os.getenv("EVAL_PRDC_K", "5")),
        )


@dataclass
class Config:
    """Main configuration container."""

    azure: AzureConfig
    processing: ProcessingConfig
    forte: ForteConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Load all configuration from environment variables."""
        return cls(
            azure=AzureConfig.from_env(),
            processing=ProcessingConfig.from_env(),
            forte=ForteConfig.from_env(),
            evaluation=EvaluationConfig.from_env(),
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config() -> Config:
    """Force reload configuration from environment."""
    global _config
    _config = Config.from_env()
    return _config
