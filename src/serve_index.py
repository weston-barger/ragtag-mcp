import logging
import subprocess
from contextlib import asynccontextmanager
from typing import Any

import chromadb
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from mcp.server.fastmcp import FastMCP

from .build_index import get_db, get_db_path
from .config_parser import RagConfig
from .llm import get_ollama_meta

logger = logging.getLogger(__name__)


def _kill_process(process: subprocess.Popen[bytes] | None) -> None:
    if not process:
        logger.info("No process to kill, returning early")
        return

    logger.info(f"Attempting to kill process {process.pid}")
    process.kill()
    logger.debug(f"Sent kill signal to process {process.pid}")
    
    try:
        logger.debug(f"Waiting for process {process.pid} to terminate (timeout: 5s)")
        process.wait(timeout=5)
        logger.info(f"Process {process.pid} terminated successfully")
    except subprocess.TimeoutExpired:
        logger.warning(f"Process {process.pid} did not terminate within timeout, force killing")
        process.kill()
        process.wait()
        logger.info(f"Process {process.pid} force killed and terminated")


@asynccontextmanager
async def lifespan(mcp):
    process = None
    try:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        yield
    finally:
        _kill_process(process)


def get_mcp(config: RagConfig) -> FastMCP[Any]:
    mcp = FastMCP("RAG", lifespan=lifespan)
    for index in config.indices:
        if not get_db(config, index).is_file():
            continue

        client = chromadb.PersistentClient(path=get_db_path(config, index))
        model = get_ollama_meta(config)
        docsearch = Chroma(
            client=client,
            collection_name=index.tool_name,
            embedding_function=model.embeddings,
        )
        qa = RetrievalQA.from_chain_type(
            llm=model.model, retriever=docsearch.as_retriever()
        )
        mcp.add_tool(
            fn=lambda prompt: qa.run(prompt),
            name=index.tool_name,
            title=index.name,
            description=index.description,
        )

    return mcp
