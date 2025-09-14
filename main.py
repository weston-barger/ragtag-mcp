#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")
import click

from src.brew_install import install_ollama
from src.build_index import build_index
from src.config_parser import ConfigParser, RagConfig
from src.serve_index import get_mcp


@click.group()
def main():
    """Main CLI tool"""
    pass


@main.command()
def osx_install():
    """install ollama on osx"""
    install_ollama(get_config())


@main.command()
def list():
    """Lists search tools in config file"""
    config = get_config()
    print("\n\n")
    print("TOOLS:\n")
    for index in config.indices:
        print(f"{index.tool_name}")
        print(f"\tName - {index.name}")
        print(f"\tDescription = {index.description}\n")


@main.command()
@click.argument("tools", nargs=-1)
@click.option("--skip-clean", is_flag=True, help="Skip cleanup of unused databases")
def build(tools, skip_clean):
    """Builds the document index"""
    build_index(get_config(), tools if len(tools) > 0 else None, clean=not skip_clean)


@main.command()
def serve():
    """Serve command"""
    mcp = get_mcp(get_config())
    mcp.run(transport="stdio")


def get_config() -> RagConfig:
    parser = ConfigParser("src/config-schema.json")
    return parser.parse_config_file("rag_config.json")


if __name__ == "__main__":
    main()
