"""Contract tests for the TOON prettier (aap_core 0.3.1) through the LlamaIndex adapter.

The TOON encoder itself lives in ``aap_core``: ``DataFrameRetriever.from_pandas``
and ``DataFrameRetriever.from_dict`` accept ``prettier="toon"`` and encode through
``toon_format`` (declared by aap_core as ``toon-format>=1.0.0``). What this module
pins down is the boundary ``aap_llamaindex`` owns, namely that a TOON payload
survives the trip

    retriever -> AgentMessage.context -> ChatCausalMultiTurnsChain prompt

verbatim. The load-bearing risk is TOON's tabular header, which is brace-bearing
(``"[2]{sku,qty}:"``), meeting ``str.format``-based template interpolation in
``_prepare_conversation`` (chain.py). Values are not re-scanned by ``str.format``,
so the payload must arrive byte-for-byte; these tests fail if that ever stops
holding.

Offline by design: the LLM is a local ``FunctionCallingLLM`` stub that records the
conversation it was handed, mirroring the ``FakeLLM`` pattern in
``tests/test_chain.py``. No network, no Ollama, no real model calls.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pytest
import toon_format
from aap_core.retriever import DataFrameRetriever
from aap_core.types import AgentMessage
from llama_index.core.base.base_retriever import (
    BaseRetriever as LlamaIndexBaseRetriever,
)
from llama_index.core.base.llms.types import LLMMetadata, MessageRole, TextBlock
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pydantic import Field, PrivateAttr

import aap_llamaindex.chain as chain_module
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from aap_llamaindex.retriever import RetrieverAdapter

# Minimum aap_core these contracts require: 0.3.1 routed prettier="toon" to
# toon_format, 0.3.2 made format_kwargs() join list-valued context into text.
TOON_MIN_AAP_CORE = (0, 3, 2)


class RecordingLLM(FunctionCallingLLM):
    """Function-calling LLM stub: records every conversation, returns fixed text.

    ``chat_with_tools`` is overridden directly because that is the single entry
    point ``ChatCausalMultiTurnsChain._generate_response`` uses; the plain
    chat/complete surfaces raise so any accidental call is loud, not silent.
    The canned response carries ``raw["usage"]``, which is where
    ``aap_llamaindex.utils.token_from_response`` reads counts from.
    """

    reply: str = Field(default="the answer")
    input_tokens: int = Field(default=21)
    output_tokens: int = Field(default=5)
    _seen: List[List[ChatMessage]] = PrivateAttr(default_factory=list)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="toon-fake")

    def chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Optional[ChatMessage] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self._seen.append([*(chat_history or []), user_msg])
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self.reply),
            raw={
                "usage": {
                    "prompt_tokens": self.input_tokens,
                    "completion_tokens": self.output_tokens,
                }
            },
        )

    @property
    def seen(self) -> List[List[ChatMessage]]:
        return self._seen

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError  # must not be called in offline tests

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def astream_complete(self, prompt: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Any = None,
        chat_history: Any = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class FakeNodeRetriever(LlamaIndexBaseRetriever):
    """LlamaIndex retriever over a fixed text list, in order. No embeddings.

    ``VectorStoreIndex`` + ``MockEmbedding`` (as used in ``test_retriever.py``)
    ties every score, so node order is arbitrary; TOON assertions need exact
    order, hence a deterministic stand-in.
    """

    def __init__(self, texts: List[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.texts = list(texts)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text=t), score=1.0) for t in self.texts]


def make_chain(
    model: Optional[FunctionCallingLLM] = None,
    user_prompt_template: str = "{query}\n{context_data}",
    system_prompt: str = "You are helpful.",
) -> ChatCausalMultiTurnsChain:
    return ChatCausalMultiTurnsChain(
        model=model if model is not None else RecordingLLM(),
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )


def toon_dataframe(records: List[Dict[str, Any]], data_key: str = "context.data"):
    """A DataFrameRetriever whose payload is TOON-encoded, via the public factory."""
    return DataFrameRetriever.from_pandas(
        pd.DataFrame(records), data_key=data_key, prettier="toon"
    )


def user_message_of(conversation: List[ChatMessage]) -> ChatMessage:
    """The current user turn: conversation[0] is the system message."""
    assert conversation[1].role == MessageRole.USER
    return conversation[1]


def user_prompt_of(chain: ChatCausalMultiTurnsChain, message: AgentMessage) -> str:
    """Rendered user prompt text handed to the model."""
    return str(user_message_of(chain._prepare_conversation(message)).content)


def resolved_aap_core_version() -> tuple:
    """Installed aap_core as a comparable tuple, tolerant of suffixes."""
    raw = version("aap_core")
    numbers: List[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    return tuple(numbers)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Keep the capability lookup inert so no test can reach the Hub."""
    monkeypatch.setattr(chain_module, "huggingface_capabilities", lambda repo_id: {})


class TestResolvedCoreSupportsToon:
    """The upgrade claim: the aap_core this repo resolves actually enables TOON."""

    def test_toon_format_distribution_is_importable(self):
        """toon_format is importable and round-trips, so the core branch can run."""
        assert toon_format.encode({"a": 1}) == "a: 1"
        assert toon_format.decode("a: 1") == {"a": 1}

    def test_aap_core_version_meets_toon_requirement(self):
        """Resolved aap_core is >= 0.3.1, the release that enabled toon_format."""
        try:
            resolved = resolved_aap_core_version()
        except PackageNotFoundError as exc:  # pragma: no cover - env misconfig
            pytest.fail(f"aap_core is not installed in this environment: {exc}")
        assert resolved >= TOON_MIN_AAP_CORE, (
            f"aap_core {resolved} resolved from the environment does not enable "
            f"toon_format; aap_core>={'.'.join(map(str, TOON_MIN_AAP_CORE))} is required. "
            "Bump the dependency floor in pyproject.toml (line 13 still says "
            "'aap_core>=0.1.0') and re-run uv sync."
        )

    def test_from_pandas_routes_toon_to_toon_format(self):
        """prettier="toon" encodes records losslessly instead of falling to tabulate."""
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        message = toon_dataframe(records).retrieve(AgentMessage(query="q"))

        payload = message.context["data"]
        assert isinstance(payload, str), (
            f"core must store one TOON string, got {type(payload).__name__}"
        )
        assert payload == "[2]{sku,qty}:\n  A-1,2\n  B-2,7"
        assert toon_format.decode(payload) == records

    def test_from_dict_routes_toon_to_toon_format(self):
        """prettier="toon" on a dict encodes it losslessly."""
        data = {"warehouse": "north", "capacity": 128}
        message = DataFrameRetriever.from_dict(data, prettier="toon").retrieve(
            AgentMessage(query="q")
        )

        assert message.context["data"] == "warehouse: north\ncapacity: 128"
        assert toon_format.decode(message.context["data"]) == data

    def test_toon_keyword_is_case_sensitive_and_degrades_silently(self):
        """Only the exact lowercase "toon" routes to toon_format.

        A misspelled prettier is passed to tabulate as a tablefmt, which does not
        reject unknown formats. This test documents the trap: a typo yields a
        plausible-looking string that is NOT TOON, with no exception raised.
        """
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        typo = DataFrameRetriever.from_pandas(pd.DataFrame(records), prettier="TOON")
        payload = typo.retrieve(AgentMessage(query="q")).context["data"]

        assert payload == "-  -\n0  0\n1  1\n-  -"
        assert "{" not in payload, (
            "prettier='TOON' unexpectedly produced TOON output; the routing "
            "condition in aap_core changed and this canary needs revisiting"
        )
        with pytest.raises(Exception):
            toon_format.decode(payload)


class TestToonPayloadIntoLlamaIndexPrompt:
    """TOON context must reach the LlamaIndex conversation verbatim."""

    def test_dataframe_toon_payload_reaches_user_message(self):
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        message = toon_dataframe(records).retrieve(AgentMessage(query="total qty?"))

        prompt = user_prompt_of(make_chain(), message)

        assert prompt == "total qty?\n" + toon_format.encode(records)

    def test_toon_tabular_header_braces_survive_str_format(self):
        """TOON's "{cols}" header must not be consumed as a format placeholder."""
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        payload = toon_format.encode(records)
        assert "{" in payload, (
            "TOON no longer emits brace-bearing headers; this regression "
            "canary no longer exercises the risk it was written for"
        )

        message = AgentMessage(query="q", context={"data": payload})
        prompt = user_prompt_of(make_chain(), message)

        assert payload in prompt, f"header was eaten by str.format: {prompt!r}"
        assert prompt.count(payload) == 1, "payload must appear exactly once"

    def test_from_dict_toon_payload_reaches_user_message(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
        message = DataFrameRetriever.from_dict(data, prettier="toon").retrieve(
            AgentMessage(query="inventory")
        )

        prompt = user_prompt_of(make_chain(), message)

        assert prompt == "inventory\n" + toon_format.encode(data)
        assert toon_format.decode(prompt.split("\n", 1)[1]) == data

    def test_toon_payload_survives_message_flattening(self):
        """to_dict()/format_kwargs() keep the TOON string intact under context_data."""
        payload = toon_format.encode([{"sku": "A-1", "qty": 2}])
        message = AgentMessage(query="q", context={"data": payload})

        assert message.format_kwargs()["context_data"] == payload
        assert message.to_dict()["context_data"] == payload

    def test_system_message_is_unaffected_by_toon_context(self):
        """The system prompt is never interpolated, so its braces stay literal."""
        payload = toon_format.encode([{"sku": "A-1", "qty": 2}])
        message = AgentMessage(query="q", context={"data": payload})
        conversation = make_chain(
            system_prompt="Rules: answer from {data} only"
        )._prepare_conversation(message)

        assert conversation[0].role == MessageRole.SYSTEM
        assert conversation[0].content == "Rules: answer from {data} only"
        assert payload in str(conversation[1].content)

    def test_user_message_carries_a_single_text_block(self):
        """Text-only turns are one TextBlock whose text equals the rendered prompt."""
        payload = toon_format.encode([{"sku": "A-1", "qty": 2}])
        message = AgentMessage(query="q", context={"data": payload})

        blocks = user_message_of(make_chain()._prepare_conversation(message)).blocks

        assert len(blocks) == 1, (
            f"expected one text block, got {[b.block_type for b in blocks]}"
        )
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == f"q\n{payload}"

    def test_toon_survives_a_multimodal_user_message(self):
        """LlamaIndex-specific: with media, the prompt is blocks[0], not content.

        The adapter assembles ``[TextBlock(prompt), ImageBlock, ...]`` when media
        is present, so a TOON table has to survive inside the text block.
        """
        payload = toon_format.encode([{"sku": "A-1", "qty": 2}])
        message = AgentMessage(
            query="stock?",
            context={"data": payload},
            query_media=[("image", "https://example.com/pallet.png")],
        )
        chain = make_chain()
        chain.media_support = {"image": "enabled"}

        user_msg = user_message_of(chain._prepare_conversation(message))

        assert [block.block_type for block in user_msg.blocks] == ["text", "image"]
        assert user_msg.blocks[0].text == f"stock?\n{payload}"
        assert payload in str(user_msg.content)

    @pytest.mark.parametrize(
        "records",
        [
            [{"sku": "A-1", "qty": 2}],
            [
                {"sku": "A-1", "qty": 2},
                {"sku": "B-2", "qty": 7},
                {"sku": "C-3", "qty": 0},
            ],
            [{"name": "café", "note": "你好"}],
            [{"id": 1, "meta": {"x": 1}}, {"id": 2, "meta": {"x": 2}}],
            [{"a": 1, "b": "line1\nline2"}],
        ],
        ids=["single-row", "multi-row", "unicode", "nested-column", "multiline-value"],
    )
    def test_various_tables_are_delivered_verbatim(self, records):
        payload = toon_format.encode(records)
        message = AgentMessage(query="q", context={"data": payload})

        prompt = user_prompt_of(make_chain(), message)

        assert prompt == "q\n" + payload
        assert toon_format.decode(prompt.split("\n", 1)[1]) == records

    def test_empty_dataframe_encodes_to_empty_table_token(self):
        """Edge case: zero rows must not blow up the prompt path."""
        retriever = DataFrameRetriever.from_pandas(
            pd.DataFrame([{"sku": "A-1", "qty": 2}]).iloc[0:0], prettier="toon"
        )
        message = retriever.retrieve(AgentMessage(query="q"))

        assert message.context["data"] == "[]"
        assert user_prompt_of(make_chain(), message) == "q\n[]"
        assert toon_format.decode("[]") == []

    def test_nan_becomes_null_and_stays_prompt_safe(self):
        """Known lossiness: NaN is encoded as TOON null, decoded as None."""
        records = [{"a": 1.0, "b": float("nan")}]
        message = toon_dataframe(records).retrieve(AgentMessage(query="q"))
        payload = message.context["data"]

        assert payload == "[1]{a,b}:\n  1,null"
        assert toon_format.decode(payload) == [{"a": 1.0, "b": None}]
        assert payload in user_prompt_of(make_chain(), message)


class TestRetrieverAdapterWithToon:
    """The LlamaIndex-side retriever must not mangle TOON payloads."""

    def test_adapter_stores_toon_encoded_node_text_verbatim(self):
        """node.get_text() is stored as-is under the default data_key."""
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        payload = toon_format.encode(records)
        adapter = RetrieverAdapter(retriever=FakeNodeRetriever([payload]))

        result = adapter.retrieve(AgentMessage(query="stock"))

        assert result.context["data"] == [payload]
        assert toon_format.decode(result.context["data"][0]) == records

    def test_adapter_keeps_multiple_toon_nodes_separate(self):
        """One TOON table per node: the adapter must not join or re-encode."""
        tables = [[{"id": 1}], [{"id": 2}, {"id": 3}]]
        payloads = [toon_format.encode(t) for t in tables]
        adapter = RetrieverAdapter(retriever=FakeNodeRetriever(payloads))

        result = adapter.retrieve(AgentMessage(query="q"))

        assert result.context["data"] == payloads
        assert [toon_format.decode(p) for p in result.context["data"]] == tables

    def test_adapter_custom_data_key_holds_toon_payload(self):
        payload = toon_format.encode([{"sku": "A-1"}])
        adapter = RetrieverAdapter(
            retriever=FakeNodeRetriever([payload]), data_key="context.stock"
        )

        result = adapter.retrieve(AgentMessage(query="q"))

        assert result.context == {"stock": [payload]}
        assert toon_format.decode(result.context["stock"][0]) == [{"sku": "A-1"}]

    def test_adapter_merges_toon_payload_into_existing_context(self):
        payload = toon_format.encode([{"sku": "A-1"}])
        adapter = RetrieverAdapter(retriever=FakeNodeRetriever([payload]))
        message = AgentMessage(query="q", context={"note": "kept"})

        result = adapter.retrieve(message)

        assert result.context == {"note": "kept", "data": [payload]}

    def test_adapter_toon_payload_is_decodable_after_prompt_interpolation(self):
        """A TOON table retrieved via a LlamaIndex retriever must stay decodable."""
        records = [{"sku": "A-1"}]
        payload = toon_format.encode(records)
        adapter = RetrieverAdapter(
            retriever=FakeNodeRetriever([payload]), data_key="context.stock"
        )
        result = adapter.retrieve(AgentMessage(query="q"))

        prompt = user_prompt_of(
            make_chain(user_prompt_template="{query}\n{context_stock}"), result
        )
        assert toon_format.decode(prompt.split("\n", 1)[1]) == records

    def test_adapter_with_toon_dataframe_post_process_merges_contexts(self):
        """Composition: LlamaIndex nodes plus a TOON table in one message context."""
        node_payload = toon_format.encode([{"chunk": "retrieved text"}])
        table_records = [{"sku": "A-1", "qty": 2}]
        adapter = RetrieverAdapter(
            retriever=FakeNodeRetriever([node_payload]),
            post_process=toon_dataframe(table_records, data_key="context.table"),
        )

        result = adapter(AgentMessage(query="q"))

        assert result.context["data"] == [node_payload]
        assert toon_format.decode(result.context["table"]) == table_records

    def test_adapter_rejects_non_prefixed_data_key_for_toon_context(self):
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=FakeNodeRetriever([]), data_key="toon_data")


class TestToonEndToEnd:
    """Retriever -> chain.invoke -> AgentMessage, with a stubbed model."""

    def test_retrieve_then_invoke_delivers_toon_to_the_model(self):
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        payload = toon_format.encode(records)
        model = RecordingLLM(reply="9 units")
        chain = make_chain(model=model)
        message = toon_dataframe(records)(AgentMessage(query="total qty?"))

        result = chain.invoke(message)

        assert str(model.seen[0][1].content) == f"total qty?\n{payload}"
        assert [m.role for m in model.seen[0][:1]] == [MessageRole.SYSTEM], (
            "only the system message may precede the current user turn"
        )
        assert result.responses == [("chain", "9 units")]
        assert result.execution_result == "success"
        assert result.token_usage["total"]["input_tokens"] == 21
        assert result.token_usage["total"]["output_tokens"] == 5
        assert result.token_usage["total"]["total_tokens"] == 26
        assert result.context["data"] == payload, "context must survive the chain"

    def test_invoke_round_trips_toon_context_back_into_agent_message(self):
        """Round-trip: TOON in context -> prompt -> out again -> decodes."""
        records = [{"sku": "A-1", "qty": 2}]
        payload = toon_format.encode(records)
        chain = make_chain(model=RecordingLLM())

        message = chain.invoke(AgentMessage(query="q", context={"data": payload}))

        assert toon_format.decode(message.context["data"]) == records
        assert toon_format.decode(message.to_dict()["context_data"]) == records

    def test_toon_context_is_not_duplicated_by_history(self):
        """TOON lands in the current user turn only, never in replayed history."""
        payload = toon_format.encode([{"sku": "A-1", "qty": 2}])
        model = RecordingLLM()
        chain = make_chain(model=model)
        chain.include_history = 2
        message = AgentMessage(
            query="q",
            context={"data": payload},
            responses=[("user", "earlier question"), ("chain", "earlier answer")],
        )

        conversation = chain._prepare_conversation(message)

        assert str(conversation[1].content) == f"q\n{payload}"
        assert [m.role for m in conversation] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        joined = "\n".join(str(m.content) for m in conversation)
        assert joined.count(payload) == 1, "TOON must appear in the current turn only"
