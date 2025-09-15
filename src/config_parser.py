import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import jsonschema
from jsonschema import validate


def _get_schema_path() -> Path:
    return Path(__file__).parent / Path("config-schema.json")


@dataclass
class ModelConfig:
    """Represents model configuration for embedding and LLM."""

    embedding: str
    llm: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary data."""
        return cls(embedding=data["embedding"], llm=data["llm"])


@dataclass
class RagIndex:
    """Represents a single RAG search index configuration."""

    name: str
    tool_name: str
    description: str
    paths: List[str]
    glob_pattern: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RagIndex":
        """Create RagIndex from dictionary data."""
        return cls(
            name=data["name"],
            tool_name=data["toolName"],
            description=data["description"],
            paths=data["paths"],
            glob_pattern=data["globPattern"],
        )


@dataclass
class RagConfig:
    """Represents the complete RAG configuration."""

    db_storage_path: str
    model: ModelConfig
    indices: List[RagIndex]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RagConfig":
        """Create RagConfig from dictionary data."""
        model = ModelConfig.from_dict(data["model"])
        indices = [RagIndex.from_dict(index_data) for index_data in data["indices"]]
        return cls(db_storage_path=data["dbStoragePath"], model=model, indices=indices)


class ConfigParser:
    """Parser for RAG configuration files."""

    def __init__(self):
        """Initialize parser with optional schema path for validation."""
        self.schema = None
        self.schema_path = str(_get_schema_path())
        self.load_schema(self.schema_path)

    def load_schema(self, schema_path: str) -> None:
        """Load JSON schema for validation."""
        with open(schema_path, "r") as f:
            self.schema = json.load(f)

    def validate_config(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        if not self.schema:
            raise ValueError("Schema not loaded. Call load_schema() first.")

        try:
            validate(instance=config_data, schema=self.schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            raise ValueError(f"Schema validation failed: {e.message}")

    def parse_config_file(self, path: str, validate_schema: bool = True) -> RagConfig:
        """Parse configuration file and return RagConfig object."""
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        if validate_schema:
            self.validate_config(config_data)

        return RagConfig.from_dict(config_data)
