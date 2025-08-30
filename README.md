# Model server

LLM inferencing with efficient memory managment, RAG implementation, LangChain, FASTAPI, FAISS, and PyTorch.

I built this just cause I couldn't make Ollama work, and llama.cpp has its own feature limitations.


## FEATURES
- Easy model extensibility
- Easy Conversation managment
- Efficient memory managment
- Large CLI
- Fast Configuration
- RAG for Browsing and Local data search
- Database Search for context awareness
- Web Interface
- server application toolkit for easy service launching

## Installation

No PyPI package available, build from source

```bash
pip install -r requirements.txt
```

Build Python Package

```bash
python3 setup.py develop
```

## LLM inferencing

```bash
model_server -start -config=".modelconf" -model="LFM2_1_2B"
```


## Build docs

```bash
cd docs
pip install -r requirements.txt
make html
```
