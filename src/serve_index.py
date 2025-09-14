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


@asynccontextmanager
def lifespan(mcp):
    process = subprocess.Popen(
        ["ollama", "serve"],
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield
    process.terminate()
    process.wait()


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
