# aap_llamaindex

LlamaIndex integration of the AI Agent Pattern framework.

![PyPI - License](https://img.shields.io/pypi/l/aap_llamaindex)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aap_llamaindex)
![PyPI - Version](https://img.shields.io/pypi/v/aap_llamaindex?label=%20)
![Python Version](https://img.shields.io/pypi/pyversions/aap_llamaindex)

## What is this?

`aap_llamaindex` is the LlamaIndex integration package within the [AI Agent Pattern](https://github.com/quanghona/agent_design_pattern) project. It bridges LlamaIndex's powerful RAG and agent capabilities with the AAP orchestration framework, enabling you to build multi-agent workflows that leverage LlamaIndex's data ingestion, retrieval, and function-calling features — all while maintaining a consistent agent interface via `aap_core`.

The package delegates LLM manipulation to LlamaIndex while reusing the core orchestration, agent abstraction, and token tracking from `aap_core`. This means you can swap between LangChain, DSPy, or LlamaIndex backends without changing your agent architecture.

## Quick Start

```bash
pip install aap_llamaindex
# or
uv add aap_llamaindex
```

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from llama_index.llms.ollama import Ollama

model = Ollama(model="llama3.2:latest", base_url="localhost:11434")
chain = ChatCausalMultiTurnsChain(
    model=model,
    system_prompt="You are a helpful assistant.",
    user_prompt_template="{query}",
)
```

## Features

- **Multi-turn Causal Chains**: `ChatCausalMultiTurnsChain` with history management, tool calling, and automatic token tracking
- **Retrieval Adapter**: `RetrieverAdapter` wrapping LlamaIndex `BaseRetriever` for seamless RAG integration with `aap_core`'s retrieval system
- **Function Calling**: Full support for LlamaIndex's `FunctionCallingLLM` tool execution within agent chains
- **Token Usage Tracking**: Automatic conversion of LlamaIndex token counts to unified `TokenUsage` objects
- **A2A Protocol Compatible**: Agents built with this package integrate with the A2A `AgentCard` protocol via `aap_core`
- **Context Passing**: Built-in support for passing final responses between chained agents as context

## Installation

### Basic

```bash
pip install aap_llamaindex
```

### With Optional Dependencies

```bash
# For Weaviate vector store support
pip install "aap_llamaindex[weaviate]"
```

### Development

```bash
uv sync --group dev
```

## Usage

### Building an Agent with a Chain

```python
from a2a.types import AgentCard, AgentSkill
from aap_core.agent import BaseAgent
from aap_core.types import AgentMessage
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from llama_index.llms.ollama import Ollama

model = Ollama(model="llama3.2:latest", base_url="localhost:11434")

class MyAgent(BaseAgent):
    chain: ChatCausalMultiTurnsChain

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        message = self.chain.invoke(message, **kwargs)
        message.execution_result = "success"
        message.origin = self.card.name
        self.state = "idle"
        return message

chain = ChatCausalMultiTurnsChain(
    model=model,
    system_prompt="You are a helpful assistant.",
    user_prompt_template="{query}",
)

agent = MyAgent(
    card=AgentCard(
        name="my-agent",
        description="A simple LlamaIndex agent",
        skills=[AgentSkill(id="default", name="default", description="default")],
    ),
    chain=chain,
)
```

### Using RAG with RetrieverAdapter

```python
from aap_llamaindex.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex

# Build an index
text_content = "Your domain-specific data here..."
document = Document(text=text_content)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents([document], embed_model=embed_model)

# Create the adapter
retriever = index.as_retriever(similarity_top_k=3)
adapter = RetrieverAdapter(retriever=retriever)

# Use it in a message
message = AgentMessage(query="Your question here")
message = adapter(message)
# Retrieved context is now in message.context
```

### Chaining Agents with Context Passing

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain

# First chain: generates code
code_chain = ChatCausalMultiTurnsChain(
    model=model,
    system_prompt="You are a coding assistant.",
    user_prompt_template="{query}",
)
code_chain.final_response_as_context("context_code")

# Second chain: uses code output as context
doc_chain = ChatCausalMultiTurnsChain(
    model=model,
    system_prompt="You are a documentation writer.",
    user_prompt_template="Input: {query}\n\nCode: {context_code}",
)
```

## Architecture

This package is part of the AI Agent Pattern monorepo. It depends on `aap_core` for the foundational agent and chain abstractions:

| Package | Description |
|---------|-------------|
| `aap_core` | Core orchestration framework, agent abstraction, RL training |
| `aap_llamaindex` | **LlamaIndex integration** (this package) |
| `aap_langchain` | LangChain integration |
| `aap_dspy` | DSPy integration |
| `aap_transformers` | Hugging Face Transformers integration |

### Key Components

- **`ChatCausalMultiTurnsChain`**: Extends `BaseCausalMultiTurnsChain` from `aap_core`. Handles multi-turn conversations with LlamaIndex `ChatMessage`/`ChatResponse` types, including tool calling and history management.
- **`RetrieverAdapter`**: Wraps LlamaIndex `BaseRetriever` to integrate with `aap_core`'s pluggable retrieval system. Automatically stores retrieved data in `AgentMessage.context`.
- **`token_from_response`**: Utility to convert LlamaIndex `ChatResponse`/`CompletionResponse` token counts into the unified `TokenUsage` format.

## Examples

Full example notebooks are available in the [example/llamaindex](https://github.com/quanghona/agent_design_pattern/tree/master/example/llamaindex) directory:

| Example | Description |
|---------|-------------|
| `simple_loop.ipynb` | Loop agent that iteratively generates responses |
| `sequential.ipynb` | Sequential multi-agent pipeline with context passing |
| `self_reflection.ipynb` | Agent that reflects on and improves its own output |
| `coordinator.ipynb` | Coordinated multi-agent collaboration |
| `debate.ipynb` | Agent debate pattern |
| `voting.ipynb` | Voting-based consensus among agents |
| `parallel.ipynb` | Parallel agent execution |
| `retrieve.ipynb` | RAG integration with RetrieverAdapter |
| `cross_reflection.ipynb` | Cross-agent reflection |
| `iterative_refinement.ipynb` | Iterative improvement pattern |

## Documentation

- Full project documentation: [agent_design_pattern docs](https://github.com/quanghona/agent_design_pattern)
- LlamaIndex docs: [docs.llamaindex.ai](https://docs.llamaindex.ai/)
- A2A Protocol: [github.com/google/a2a-sdk](https://github.com/google/a2a-sdk)

## Contributing

See [CONTRIBUTING.md](https://github.com/quanghona/agent_design_pattern) for details on setting up the development environment and contributing.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/quanghona/agent_design_pattern/blob/master/LICENSE) file for details.
