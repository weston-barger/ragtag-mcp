import subprocess

from .config_parser import RagConfig


def brew_install(package: str) -> str:
    """Install a package using brew."""
    try:
        result = subprocess.run(
            ["brew", "install", package], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to install {package}: {e.stderr}")


def model_install(model: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "pull", model], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to install Ollama model {model}: {e.stderr}")


def install_ollama(config: RagConfig) -> None:
    stdout = brew_install("ollama")
    print(stdout)

    for model in [config.model.embedding, config.model.llm]:
        stdout = model_install(model)
        print(stdout)
