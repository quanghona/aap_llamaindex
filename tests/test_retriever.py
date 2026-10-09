"""Test cases for the RetrieverAdapter module in aap_llamaindex."""

import pytest
from aap_core.types import AgentMessage
from aap_llamaindex.retriever import RetrieverAdapter
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

_TEST_EMBED_MODEL = MockEmbedding(embed_dim=1536)


def _create_test_documents():
    """Create simple test documents for testing the retriever."""
    return [
        Document(text="The capital of France is Paris."),
        Document(text="Python is a popular programming language."),
        Document(text="Machine learning is a subset of artificial intelligence."),
    ]


def _create_index_with_documents(documents):
    """Create a VectorStoreIndex from documents and return its retriever."""
    index = VectorStoreIndex.from_documents(documents, embed_model=_TEST_EMBED_MODEL)
    return index.as_retriever()


class TestRetrieverAdapterDataKeyValidation:
    """Test data_key field validation."""

    def test_valid_data_key_default(self):
        """Test creating a retriever with default data_key."""
        llamaindex_retriever = _create_index_with_documents(_create_test_documents())
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        assert adapter.data_key == "context.data"

    def test_valid_data_key_custom(self):
        """Test creating a retriever with a custom valid data_key."""
        llamaindex_retriever = _create_index_with_documents(_create_test_documents())
        adapter = RetrieverAdapter(
            retriever=llamaindex_retriever, data_key="context.custom_field"
        )
        assert adapter.data_key == "context.custom_field"

    def test_valid_data_key_nested(self):
        """Test creating a retriever with a nested custom data_key."""
        llamaindex_retriever = _create_index_with_documents(_create_test_documents())
        adapter = RetrieverAdapter(
            retriever=llamaindex_retriever, data_key="context.retrieval.docs"
        )
        assert adapter.data_key == "context.retrieval.docs"

    def test_invalid_data_key_no_prefix(self):
        """Test that data_key without 'context.' prefix raises ValueError."""
        llamaindex_retriever = _create_index_with_documents(_create_test_documents())
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=llamaindex_retriever, data_key="data")

    def test_invalid_data_key_wrong_prefix(self):
        """Test that data_key with wrong prefix raises ValueError."""
        llamaindex_retriever = _create_index_with_documents(_create_test_documents())
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=llamaindex_retriever, data_key="message.data")

    def test_invalid_data_key_empty(self):
        """Test that empty data_key raises ValueError."""
        llamaindex_retriever = _create_index_with_documents(_create_test_documents())
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=llamaindex_retriever, data_key="")


class TestRetrieverAdapterRetrieve:
    """Test retrieve method with real LlamaIndex retriever."""

    def test_retrieve_returns_message_with_context(self):
        """Test that retrieve adds retrieved documents to message context."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        # MockEmbedding gives every document the same vector, so ranking and
        # top-k selection are arbitrary. Retrieve all documents to make the
        # content assertion deterministic; position assertions are meaningless.
        llamaindex_retriever.similarity_top_k = len(documents)
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        message = AgentMessage(query="What is the capital of France?")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert "data" in result.context
        assert isinstance(result.context["data"], list)
        assert len(result.context["data"]) == len(documents)
        assert any("Paris" in text for text in result.context["data"])

    def test_retrieve_default_data_key(self):
        """Test that default data_key is 'context.data' and stored as 'data'."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        message = AgentMessage(query="test query")

        result = adapter.retrieve(message)

        assert result.context == {"data": result.context["data"]}
        assert "data" in result.context

    def test_retrieve_custom_data_key(self):
        """Test that custom data_key is correctly stripped of 'context.' prefix."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        adapter = RetrieverAdapter(
            retriever=llamaindex_retriever,
            data_key="context.custom_docs",
        )
        message = AgentMessage(query="test query")

        result = adapter.retrieve(message)

        assert "custom_docs" in result.context
        assert isinstance(result.context["custom_docs"], list)

    def test_retrieve_merges_with_existing_context(self):
        """Test that retrieve merges with existing message context."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        message = AgentMessage(
            query="test query",
            context={"existing_key": "existing_value"},
        )

        result = adapter.retrieve(message)

        assert "existing_key" in result.context
        assert result.context["existing_key"] == "existing_value"
        assert "data" in result.context

    def test_retrieve_overwrites_existing_data_key(self):
        """Test that retrieve overwrites existing data under the same key."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        message = AgentMessage(
            query="test query",
            context={"data": ["Old doc"]},
        )

        result = adapter.retrieve(message)

        assert result.context["data"] != ["Old doc"]
        assert isinstance(result.context["data"], list)
        assert len(result.context["data"]) > 0

    def test_retrieve_multiple_nodes(self):
        """Test that retrieve returns multiple nodes when available."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        message = AgentMessage(query="test query")

        result = adapter.retrieve(message)

        assert isinstance(result.context["data"], list)
        assert len(result.context["data"]) >= 1

    def test_retrieve_preserves_query(self):
        """Test that retrieve preserves the original query in the message."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        original_query = "What is machine learning?"
        message = AgentMessage(query=original_query)

        result = adapter.retrieve(message)

        assert result.query == original_query

    def test_retrieve_content_matches_documents(self):
        """Test that retrieved content matches the original document text."""
        documents = _create_test_documents()
        llamaindex_retriever = _create_index_with_documents(documents)
        # MockEmbedding ties every score, so top-k selection is arbitrary;
        # retrieve all documents to make the content assertion deterministic.
        llamaindex_retriever.similarity_top_k = len(documents)
        adapter = RetrieverAdapter(retriever=llamaindex_retriever)
        message = AgentMessage(query="artificial intelligence")

        result = adapter.retrieve(message)

        retrieved_text = " ".join(result.context["data"])
        assert (
            "artificial intelligence" in retrieved_text.lower()
            or "machine learning" in retrieved_text.lower()
        )
