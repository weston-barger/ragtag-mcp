from dataclasses import dataclass

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from .config_parser import RagConfig

OLLAMA_URL: str = "http://127.0.0.1:11434"


@dataclass
class OllamaMeta:
    embeddings: OllamaEmbeddings
    model: OllamaLLM


def get_ollama_meta(config: RagConfig) -> OllamaMeta:
    return OllamaMeta(
        embeddings=OllamaEmbeddings(model=config.model.embedding, base_url=OLLAMA_URL),
        model=OllamaLLM(model=config.model.llm, base_url=OLLAMA_URL),
    )
