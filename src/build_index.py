import os
import shutil
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

from .config_parser import RagConfig, RagIndex
from .llm import OllamaMeta, get_ollama_meta


def get_db_path(config: RagConfig, index: RagIndex) -> Path:
    return Path(config.db_storage_path) / Path(index.tool_name)


def get_db(config: RagConfig, index: RagIndex) -> Path:
    return get_db_path(config, index) / Path("chroma.sqlite3")


def get_current_dbs(config: RagConfig) -> list[str]:
    return [
        item.name for item in Path(config.db_storage_path).iterdir() if item.is_dir()
    ]


def confirm_database_removal(db: str, db_path: Path) -> bool:
    response = None
    while response not in ["y", "n", "yes", "no"]:
        response = (
            input(f'"{db}" ({db_path}) not found in config. Ready to remove it? (Y/n)')
            .strip()
            .lower()
        )
    return response in ("yes", "y")


def cleanup_unused_databases(
    config: RagConfig, all_tools: dict[str, RagIndex], clean: bool
) -> None:
    if not clean:
        return

    dbs = get_current_dbs(config)
    for db in dbs:
        if db not in all_tools:
            db_path = Path(config.db_storage_path) / Path(db)

            if confirm_database_removal(db, db_path):
                print(f"Removing {db_path}...")
                shutil.rmtree(db_path)


def get_indices_to_process(
    config: RagConfig, tools: list[str] | None, all_tools: dict[str, RagIndex]
) -> list[RagIndex]:
    if tools is None:
        return config.indices

    indices = []
    for tool in tools:
        if tool not in all_tools:
            raise ValueError(f"Tool {tool} not found in config.")
        indices.append(all_tools[tool])
    return indices


def load_documents_from_paths(index: RagIndex) -> list:
    data = []
    for index_path in index.paths:
        print(f"Indexing path: {index_path}:")
        for pattern in index.glob_pattern:
            print(f"Globbing: {pattern}")
            loader_cls = (
                UnstructuredFileLoader
                if not pattern.endswith("md")
                else UnstructuredMarkdownLoader
            )
            loader = DirectoryLoader(
                path=index_path,
                glob=pattern,
                loader_cls=loader_cls,
                show_progress=True,
                recursive=True,
            )
            data.extend(loader.load())
    return data


def create_vector_index(
    documents: list, config: RagConfig, index: RagIndex, ollama_meta: OllamaMeta
) -> None:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    db_path = get_db_path(config, index)

    print(f'\nBuilding index for tool "{index.tool_name}". Please wait.')
    Chroma.from_documents(
        texts,
        ollama_meta.embeddings,
        collection_name=index.tool_name,
        persist_directory=str(db_path),
    )
    print(f'Finished building index for tool "{index.tool_name}".\n\n')


def build_index(
    config: RagConfig, tools: list[str] | None = None, clean: bool = True
) -> None:
    ollama_meta = get_ollama_meta(config)
    all_tools = {index.tool_name: index for index in config.indices}

    cleanup_unused_databases(config, all_tools, clean)

    indices = get_indices_to_process(config, tools, all_tools)

    for index in indices:
        documents = load_documents_from_paths(index)
        create_vector_index(documents, config, index, ollama_meta)
