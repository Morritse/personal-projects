# Ollama Project

LLM application with embedding capabilities using the Ollama framework.

## Structure

```
ollama/
├── create_embeddings_ollama.ipynb  # Embedding generation
├── query_with_embeddings.py        # Query processing
├── vector_processor.py             # Vector operations
├── minima/                        # Minima integration
└── ollama/                        # Ollama core
```

## Components

### Embedding System
- Vector embedding generation
- Query processing
- Vector operations

### Integration
- Minima framework integration
- Docker containerization
- Local LLM deployment

### Core Features
- Model integration
- Query handling
- Resource management

## Setup

Required packages:
```
numpy
torch
transformers
```

## Usage

Generate embeddings:
```bash
python vector_processor.py
```

Process queries:
```bash
python query_with_embeddings.py
```

## Documentation

Additional documentation available in:
- minima/README.md: Minima integration guide
- ollama/README.md: Core implementation details
