"""Offline structural tests for aap_llamaindex.chain image input support.

No network or model calls: tests exercise _prepare_conversation assembly,
capability detection from model metadata (Ollama server capabilities, Hugging
Face model cards), and the media gate only. The Hugging Face lookup is stubbed
by an autouse fixture so detection is deterministic offline.
"""

import base64
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest
from aap_core.types import AgentMessage, ContentType
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from pydantic import Field, PrivateAttr

import aap_llamaindex.chain as chain_module
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from aap_llamaindex.utils import media_ref_to_image_block

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Capability evidence comes from metadata, so tests declare it explicitly
# instead of relying on model names.
VISION_CAPS = ["completion", "tools", "vision"]
TEXT_ONLY_CAPS = ["completion", "tools"]


class FakeOllamaClient:
    """Stand-in for the ollama-python Client: records show() calls."""

    def __init__(self, capabilities: Any, raises: bool = False) -> None:
        self.capabilities = capabilities
        self.raises = raises
        self.calls: List[str] = []

    def show(self, name: str) -> Any:
        self.calls.append(name)
        if self.raises:
            raise ConnectionError("ollama server unreachable")
        return SimpleNamespace(capabilities=self.capabilities)


class FakeLLM(FunctionCallingLLM):
    """Minimal function-calling LLM stub; never invoked by these tests.

    Mirrors the surfaces detection consults: an ollama-like `model` name and
    `client.show()` returning server-reported capabilities. show_capabilities
    None means the integration has no such client (like the OpenAI one).
    metadata_raises simulates integrations whose metadata property probes a
    live server: detection must not break on it.
    """

    model: str = Field(default="fake-model")
    show_capabilities: Optional[List[str]] = Field(default=None)
    show_raises: bool = Field(default=False)
    metadata_raises: bool = Field(default=False)
    _fake_client: Any = PrivateAttr(default=None)

    @property
    def client(self) -> Any:
        if self.show_capabilities is None and not self.show_raises:
            return None
        if self._fake_client is None:
            self._fake_client = FakeOllamaClient(
                self.show_capabilities, self.show_raises
            )
        return self._fake_client

    @property
    def metadata(self) -> LLMMetadata:
        if self.metadata_raises:
            raise ConnectionError("simulated server probe failure")
        return LLMMetadata(model_name=self.model)

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
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


def make_chain(
    model_name: str = "fake-model",
    capabilities: Optional[List[str]] = None,
    show_raises: bool = False,
    **kwargs,
) -> ChatCausalMultiTurnsChain:
    return ChatCausalMultiTurnsChain(
        model=FakeLLM(
            model=model_name,
            show_capabilities=capabilities,
            show_raises=show_raises,
        ),
        system_prompt="You are helpful.",
        user_prompt_template="{query}",
        **kwargs,
    )


class HuggingFaceStub:
    """Stand-in for the model-card lookup: canned results plus call log."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[ContentType, bool]] = {}
        self.calls: List[str] = []

    def __call__(self, repo_id: str) -> Dict[ContentType, bool]:
        self.calls.append(repo_id)
        return self.results.get(repo_id, {})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Keep the default lookup inert so no test can reach the Hub."""
    monkeypatch.setattr(chain_module, "huggingface_capabilities", lambda repo_id: {})


@pytest.fixture
def hf(monkeypatch):
    """Install a recording Hugging Face stub over no_network's default."""
    stub = HuggingFaceStub()
    monkeypatch.setattr(chain_module, "huggingface_capabilities", stub)
    return stub


def user_message_of(conversation: List[ChatMessage]) -> ChatMessage:
    return conversation[1]  # [0] is the system message


class TestPrepareConversationTextOnly:
    """Regression: text-only behavior must be identical to before media support."""

    def test_plain_text_message(self):
        chain = make_chain()
        conversation = chain._prepare_conversation(AgentMessage(query="hello"))
        assert conversation[0].content == "You are helpful."
        assert user_message_of(conversation).content == "hello"
        assert isinstance(user_message_of(conversation).blocks[0], TextBlock)

    def test_template_with_context(self):
        chain = ChatCausalMultiTurnsChain(
            model=FakeLLM(),
            system_prompt="s",
            user_prompt_template="{query} about {context_topic}",
        )
        msg = AgentMessage(query="q", context={"topic": "t"})
        assert user_message_of(chain._prepare_conversation(msg)).content == "q about t"

    def test_history_appended_after_user_message(self):
        chain = make_chain()
        chain.include_history = 2
        msg = AgentMessage(
            query="now", responses=[("user", "before"), ("agent", "answer")]
        )
        conversation = chain._prepare_conversation(msg)
        assert len(conversation) == 4
        assert conversation[2].role == MessageRole.USER
        assert conversation[2].content == "before"
        assert conversation[3].content == "answer"


class TestPrepareConversationWithImages:
    """Image media must become native blocks on the current user message."""

    def test_url_image_becomes_image_block(self):
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        msg = AgentMessage(
            query="what is this?",
            query_media=[("image", "https://example.com/cat.png")],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "what is this?"
        assert isinstance(blocks[1], ImageBlock)
        assert str(blocks[1].url) == "https://example.com/cat.png"

    def test_local_path_image_becomes_base64_block(self, tmp_path):
        f = tmp_path / "cat.png"
        f.write_bytes(PNG_MAGIC + b"\x00" * 16)
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        msg = AgentMessage(query="describe", query_media=[("image", str(f))])
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert isinstance(blocks[1], ImageBlock)
        assert blocks[1].image_mimetype == "image/png"
        # ImageBlock stores the payload base64-encoded internally
        assert base64.b64decode(blocks[1].image) == f.read_bytes()

    def test_multiple_images_preserve_order(self):
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        msg = AgentMessage(
            query="two?",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("image", "https://example.com/b.jpg"),
            ],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert [str(b.url) for b in blocks[1:]] == [
            "https://example.com/a.png",
            "https://example.com/b.jpg",
        ]

    def test_no_repr_leak_into_text_block(self):
        """The core leak guard: media must never appear in the interpolated text."""
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        msg = AgentMessage(
            query="what?",
            query_media=[("image", "https://example.com/cat.png")],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert blocks[0].text == "what?"
        assert "example.com" not in blocks[0].text

    def test_images_attach_only_to_current_user_message(self):
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        chain.include_history = 1
        msg = AgentMessage(
            query="new",
            query_media=[("image", "https://example.com/x.png")],
            responses=[("user", "old text turn")],
        )
        conversation = chain._prepare_conversation(msg)
        assert any(
            isinstance(b, ImageBlock) for b in user_message_of(conversation).blocks
        )
        assert conversation[2].content == "old text turn"  # history stays text


class TestCapabilityDetection:
    """Metadata sources for the image modality gate: server capabilities, HF cards."""

    def test_ollama_reports_vision_capable(self):
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        assert chain.detect_capabilities() == {"image": True}

    def test_ollama_capability_list_without_vision_denies_image(self):
        """A non-empty list is complete: absence of vision means no image input."""
        chain = make_chain(model_name="llama3.2:latest", capabilities=TEXT_ONLY_CAPS)
        assert chain.detect_capabilities() == {"image": False}

    def test_ollama_capabilities_are_case_insensitive(self):
        chain = make_chain(capabilities=["Completion", "Vision"])
        assert chain.detect_capabilities() == {"image": True}

    def test_ollama_show_is_asked_for_the_exact_revision(self):
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        chain.detect_capabilities()
        assert chain.model.client.calls == ["gemma3:12b"]

    def test_unreachable_ollama_server_is_undecidable(self):
        """Detection must not break the request path when the server is down."""
        chain = make_chain(model_name="gemma3:12b", show_raises=True)
        assert chain.detect_capabilities() == {}

    def test_server_without_capabilities_field_is_undecidable(self):
        """Older servers report no capabilities list: no opinion, fail loud."""
        chain = make_chain(capabilities=[])
        assert chain.detect_capabilities() == {}

    def test_models_without_ollama_client_are_undecidable_offline(self, hf):
        chain = make_chain(model_name="gpt-4o")
        assert chain.detect_capabilities() == {}
        assert hf.calls == []  # provider names are not repo ids

    def test_huggingface_repo_id_models_are_looked_up_once(self, hf):
        hf.results["org/repo"] = {"image": True, "audio": False}
        chain = make_chain(model_name="org/repo")
        assert chain.detect_capabilities() == {"image": True, "audio": False}
        chain.detect_capabilities()
        assert hf.calls == ["org/repo"]

    def test_huggingface_fills_gaps_left_by_a_partial_server_report(self, hf):
        """Server reports image only; the card supplies the other modalities."""
        hf.results["org/repo"] = {"audio": True}
        chain = make_chain(model_name="org/repo", capabilities=VISION_CAPS)
        assert chain.detect_capabilities() == {"image": True, "audio": True}

    def test_cache_invalidated_when_model_is_swapped(self, hf):
        chain = make_chain(model_name="org/a")
        chain.detect_capabilities()
        chain.model = FakeLLM(model="org/b")
        chain.detect_capabilities()
        assert hf.calls == ["org/a", "org/b"]

    def test_metadata_probe_failure_does_not_break_detection(self):
        """Ollama-style metadata hits the network; detection must not depend on it."""
        chain = make_chain()
        chain.model = FakeLLM(
            model="gemma3:12b",
            show_capabilities=VISION_CAPS,
            metadata_raises=True,
        )
        assert chain.detect_capabilities() == {"image": True}


class TestGateIntegration:
    """End-to-end of the core gate through the llamaindex funnel."""

    def test_text_only_model_drops_images_with_warning(self, caplog):
        chain = make_chain(model_name="llama3.2:latest", capabilities=TEXT_ONLY_CAPS)
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert len(blocks) == 1  # text-only degradation, no exception
        assert blocks[0].text == "look"
        assert "does not support image input" in caplog.text

    def test_disabled_policy_overrides_vision_detection(self):
        chain = make_chain(
            model_name="gemma3:12b",
            capabilities=VISION_CAPS,
            media_support={"image": "disabled"},
        )
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert len(blocks) == 1
        assert not any(isinstance(b, ImageBlock) for b in blocks)

    def test_enabled_policy_forces_blocks_on_unknown_model(self):
        chain = make_chain(
            model_name="some-brand-new-model", media_support={"image": "enabled"}
        )
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, ImageBlock) for b in blocks)

    def test_undecidable_model_attaches_blocks_fail_loud(self):
        """Unknown model + image media: attach and let the API error surface."""
        chain = make_chain(model_name="some-brand-new-model")
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, ImageBlock) for b in blocks)

    def test_missing_local_image_raises(self, tmp_path):
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        msg = AgentMessage(
            query="look", query_media=[("image", str(tmp_path / "nope.png"))]
        )
        with pytest.raises(FileNotFoundError):
            chain._prepare_conversation(msg)


class TestMediaRefToImageBlock:
    """Direct tests of the MediaReference -> ImageBlock mapping."""

    def test_url_ref(self):
        block = media_ref_to_image_block(
            {
                "content_type": "image",
                "kind": "url",
                "value": "http://x/a.png",
                "mime_type": "image/png",
            }
        )
        assert isinstance(block, ImageBlock)
        assert str(block.url) == "http://x/a.png"

    def test_base64_ref(self):
        block = media_ref_to_image_block(
            {
                "content_type": "image",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "image/jpeg",
            }
        )
        assert isinstance(block, ImageBlock)
        assert block.image_mimetype == "image/jpeg"
        assert base64.b64decode(block.image) == b"\x00\x00\x00"

    def test_non_image_ref_rejected(self):
        with pytest.raises(ValueError, match="only image refs"):
            media_ref_to_image_block(
                {
                    "content_type": "audio",
                    "kind": "base64",
                    "value": "AAAA",
                    "mime_type": "audio/wav",
                }
            )
