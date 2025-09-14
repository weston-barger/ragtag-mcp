# ragtag-mcp: a local RAG setup for AI tools 
Hacky MVP for local RAG search.

# How to install

### Clone repo
Clone this repo

```
cd /Users/suhdude/repos
gh repo clone weston-barger/ragtag-mcp
```

### Pull down the codebases and documentation you want to index
You'll need local copies of the content you want to search. Example:

```
cd /Users/suhdude/repos
gh repo clone holoviz/param
gh repo clone holoviz/panel
gh repo clone bokeh/bokeh
```

### Set up your rag_config.json file 
Set up your `rag_config.json` to look like this

```json
{
  "indices": [],
  "dbStoragePath": "/Users/suhdude/repos/ragtag-mcp/db",
  "model": {
    "embedding": "nomic-embed-text:latest",
    "llm": "qwen3:latest"
  }
}
```
the `dbStoragePath` is where your vector database for searching will live. 

### Set up vitrual environment and install requirements
You'll want to set up a python venv for this repo. Run 
```
python3 -m venv .venv
source ./.venv/bin/activate
pip3 install -r requirements.txt --upgrade
```

### Install Ollama + accompanying models
The `main.py` has a command that will help you do this on OSX. Run 

```
python main.py osx_install
```

NOTE: you'll want to re-run the osx_install command if you change the embedding or llm models in the config. 

### Create your RAG search indices
Now fill our the `indices` section of your `rag_config.json` file. There is an example in `example_rag_config.json`. Then run 

```
python main.py build
```

This may take a a bit. See `python main.py build --help` to see how to build one index at a time. 

# Claude instructions
### Install the MCP server in your Claude project

Add

```
{
  "mcpServers": {
    "RAG": {
      "command": "/Users/suhdude/repos/ragtag-mcp/.venv/bin/python3",
      "args": [
        "/Users/suhdude/repos/ragtag-mcp/main.py",
        "serve"
      ]
    }
  }
}
```

to your `.mcp.json` file for your project. 

### Check install

Open claude an issue `/mcp`. You should see `RAG`!

### Optional: Tell Claude to use the RAG search to get context

Add something to your `CLAUDE.md` file to tell it to use RAG to get context

```markdown
(My project) relies on packages pkg1, pkg2, pkg3. Use the RAG search MCP server to understand how to use packages pkg1, pkg2, pkg3 when planning code changes. 
```