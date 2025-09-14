import subprocess

from .config_parser import RagConfig


def _brew_install(package: str) -> str:
    """Install a package using brew."""
    try:
        result = subprocess.run(["brew", "install", package], text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to install {package}: {e.stderr}")


def _model_install(model: str) -> str:
    try:
        result = subprocess.run(["ollama", "pull", model], text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to install Ollama model {model}: {e.stderr}")


def install_ollama(config: RagConfig) -> None:
    _brew_install("ollama")

    for model in [config.model.embedding, config.model.llm]:
        _model_install(model)
