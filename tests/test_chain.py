"""Offline structural tests for aap_llamaindex.chain image, audio and video input support.

No network or model calls: tests exercise _prepare_conversation assembly,
capability detection from model metadata (Ollama server capabilities, Hugging
Face model cards, the adapter transport ceiling), and the media gate only. The
Hugging Face lookup is stubbed by an autouse fixture so detection is
deterministic offline.
"""

import ast
import base64
import inspect
import logging
import re
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest
from aap_core.types import AgentMessage, ContentType
from aap_core.utils import SUPPORTED_MIME_TYPES
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
    VideoBlock,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from pydantic import Field, PrivateAttr

import aap_llamaindex.chain as chain_module
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from aap_llamaindex.utils import (
    _AUDIO_FORMATS,
    _bedrock_carries,
    _ollama_carries,
    _openai_carries,
    adapter_capabilities,
    media_ref_to_audio_block,
    media_ref_to_image_block,
    media_ref_to_video_block,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WAV_MAGIC = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 16
MP3_MAGIC = b"\xff\xfb\x90\x00" + b"\x00" * 16
# ISO-BMFF: box size, "ftyp", major brand "isom" -> the core probe says video/mp4.
MP4_MAGIC = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 4

# Capability evidence comes from metadata, so tests declare it explicitly
# instead of relying on model names.
VISION_CAPS = ["completion", "tools", "vision"]
AUDIO_CAPS = ["completion", "tools", "audio"]
MULTIMODAL_CAPS = ["completion", "tools", "vision", "audio"]
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


class FakeOpenAILLM(FakeLLM):
    """FakeLLM claiming the OpenAI adapter's module, for ceiling dispatch."""


FakeOpenAILLM.__module__ = "llama_index.llms.openai.base"


class FakeBedrockLLM(FakeLLM):
    """FakeLLM claiming the Bedrock Converse adapter's module, for dispatch."""


FakeBedrockLLM.__module__ = "llama_index.llms.bedrock_converse.base"


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


class TestPrepareConversationWithAudio:
    """Audio media must reach the model as audio, never as a transcript."""

    def test_url_audio_becomes_audio_block(self):
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(
            query="what is being said?",
            query_media=[("audio", "https://example.com/clip.wav")],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "what is being said?"
        assert isinstance(blocks[1], AudioBlock)
        assert str(blocks[1].url) == "https://example.com/clip.wav"

    def test_local_path_audio_becomes_base64_block(self, tmp_path):
        f = tmp_path / "clip.wav"
        f.write_bytes(WAV_MAGIC)
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert isinstance(blocks[1], AudioBlock)
        # AudioBlock stores the payload base64-encoded internally
        assert base64.b64decode(blocks[1].audio) == f.read_bytes()
        assert blocks[1].format == "wav"

    def test_audio_format_is_sniffed_not_taken_from_the_extension(self, tmp_path):
        """A mislabelled .txt carrying WAV bytes must still deliver format wav."""
        f = tmp_path / "clip.txt"
        f.write_bytes(WAV_MAGIC)
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert blocks[1].format == "wav"

    def test_mp3_maps_to_the_mp3_format_token(self, tmp_path):
        """OpenAI's input_audio enum accepts wav|mp3, so audio/mpeg must not ship as 'mpeg'."""
        f = tmp_path / "song.dat"
        f.write_bytes(MP3_MAGIC)
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert blocks[1].format == "mp3"

    def test_multiple_audio_preserve_order(self):
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(
            query="two?",
            query_media=[
                ("audio", "https://example.com/a.mp3"),
                ("audio", "https://example.com/b.wav"),
            ],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert [str(b.url) for b in blocks[1:]] == [
            "https://example.com/a.mp3",
            "https://example.com/b.wav",
        ]

    def test_images_and_audio_both_attach(self):
        chain = make_chain(model_name="gemma4-audio", capabilities=MULTIMODAL_CAPS)
        msg = AgentMessage(
            query="see and hear?",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("audio", "https://example.com/a.wav"),
            ],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert [b.block_type for b in blocks] == ["text", "image", "audio"]

    def test_no_repr_leak_into_text_block(self):
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(
            query="what?", query_media=[("audio", "https://example.com/a.wav")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert blocks[0].text == "what?"
        assert "example.com" not in blocks[0].text

    def test_audio_attaches_only_to_current_user_message(self):
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        chain.include_history = 1
        msg = AgentMessage(
            query="new",
            query_media=[("audio", "https://example.com/a.wav")],
            responses=[("user", "old text turn")],
        )
        conversation = chain._prepare_conversation(msg)
        assert any(
            isinstance(b, AudioBlock) for b in user_message_of(conversation).blocks
        )
        assert conversation[2].content == "old text turn"  # history stays text


class TestPrepareConversationWithVideo:
    """Video media must reach the model as one container part, never frames."""

    def test_url_video_becomes_video_block(self, hf):
        hf.results["org/repo"] = {"video": True}
        chain = make_chain(model_name="org/repo")
        msg = AgentMessage(
            query="what happens?",
            query_media=[("video", "https://example.com/clip.mp4")],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "what happens?"
        assert isinstance(blocks[1], VideoBlock)
        assert str(blocks[1].url) == "https://example.com/clip.mp4"

    def test_local_path_video_becomes_base64_block(self, tmp_path, hf):
        hf.results["org/repo"] = {"video": True}
        f = tmp_path / "clip.mp4"
        f.write_bytes(MP4_MAGIC + b"\x00" * 16)
        chain = make_chain(model_name="org/repo")
        msg = AgentMessage(query="watch", query_media=[("video", str(f))])
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert isinstance(blocks[1], VideoBlock)
        assert blocks[1].video_mimetype == "video/mp4"
        # VideoBlock stores the payload base64-encoded internally
        assert base64.b64decode(blocks[1].video) == f.read_bytes()
        # frame sampling is the provider's decision, never ours
        assert blocks[1].fps is None

    def test_multiple_videos_preserve_order(self):
        chain = make_chain(media_support={"video": "enabled"})
        msg = AgentMessage(
            query="two?",
            query_media=[
                ("video", "https://example.com/a.mp4"),
                ("video", "https://example.com/b.mov"),
            ],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert [str(b.url) for b in blocks[1:]] == [
            "https://example.com/a.mp4",
            "https://example.com/b.mov",
        ]

    def test_images_audio_and_video_all_attach_grouped_by_modality(self, hf):
        hf.results["org/repo"] = {"image": True, "audio": True, "video": True}
        chain = make_chain(model_name="org/repo")
        msg = AgentMessage(
            query="see hear watch?",
            query_media=[
                ("video", "https://example.com/a.mp4"),
                ("image", "https://example.com/a.png"),
                ("audio", "https://example.com/a.wav"),
            ],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert [b.block_type for b in blocks] == ["text", "image", "audio", "video"]

    def test_no_repr_leak_into_text_block(self):
        chain = make_chain(media_support={"video": "enabled"})
        msg = AgentMessage(
            query="what?", query_media=[("video", "https://example.com/a.mp4")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert blocks[0].text == "what?"
        assert "example.com" not in blocks[0].text

    def test_video_attaches_only_to_current_user_message(self):
        chain = make_chain(media_support={"video": "enabled"})
        chain.include_history = 1
        msg = AgentMessage(
            query="new",
            query_media=[("video", "https://example.com/a.mp4")],
            responses=[("user", "old text turn")],
        )
        conversation = chain._prepare_conversation(msg)
        assert any(
            isinstance(b, VideoBlock) for b in user_message_of(conversation).blocks
        )
        assert conversation[2].content == "old text turn"  # history stays text

    def test_video_denied_by_card_drops_with_warning(self, caplog, hf):
        chain = make_chain(model_name="org/repo")
        hf.results["org/repo"] = {"video": False}
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert len(blocks) == 1  # text-only degradation, no exception
        assert blocks[0].text == "watch"
        assert "does not support video input" in caplog.text

    def test_enabled_policy_forces_video_parts_on_unknown_model(self):
        chain = make_chain(
            model_name="some-brand-new-model", media_support={"video": "enabled"}
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, VideoBlock) for b in blocks)

    def test_undecidable_video_attaches_fail_loud(self):
        """Unknown model + video media: attach and let the API error surface."""
        chain = make_chain(model_name="some-brand-new-model")
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, VideoBlock) for b in blocks)

    def test_missing_local_video_raises(self, tmp_path):
        chain = make_chain(media_support={"video": "enabled"})
        msg = AgentMessage(
            query="watch", query_media=[("video", str(tmp_path / "nope.mp4"))]
        )
        with pytest.raises(FileNotFoundError):
            chain._prepare_conversation(msg)


class TestCapabilityDetection:
    """Metadata sources for the media gate: server capabilities, HF cards, ceiling."""

    def test_ollama_reports_vision_capable(self):
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        assert chain.detect_capabilities() == {"image": True, "audio": False}

    def test_ollama_reports_audio_capable(self):
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        assert chain.detect_capabilities() == {"image": False, "audio": True}

    def test_ollama_capability_list_without_vision_denies_image(self):
        """A non-empty list is complete: absence of vision means no image input."""
        chain = make_chain(model_name="llama3.2:latest", capabilities=TEXT_ONLY_CAPS)
        assert chain.detect_capabilities() == {"image": False, "audio": False}

    def test_ollama_capabilities_are_case_insensitive(self):
        chain = make_chain(capabilities=["Completion", "Vision", "Audio"])
        assert chain.detect_capabilities() == {"image": True, "audio": True}

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
        """Server opines on image/audio only; the card supplies video."""
        hf.results["org/repo"] = {"video": True}
        chain = make_chain(model_name="org/repo", capabilities=VISION_CAPS)
        assert chain.detect_capabilities() == {
            "image": True,
            "audio": False,
            "video": True,
        }

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
        assert chain.detect_capabilities() == {"image": True, "audio": False}


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

    def test_audio_denied_by_card_drops_with_warning(self, caplog, hf):
        chain = make_chain(model_name="org/repo")
        hf.results["org/repo"] = {"audio": False}
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert len(blocks) == 1  # text-only degradation, no exception
        assert blocks[0].text == "listen"
        assert "does not support audio input" in caplog.text

    def test_gate_is_per_modality_for_audio(self):
        """Audio support says nothing about image support, and vice versa."""
        chain = make_chain(
            model_name="gemma4-audio",
            capabilities=MULTIMODAL_CAPS,
            media_support={"image": "enabled", "audio": "disabled"},
        )
        msg = AgentMessage(
            query="both?",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("audio", "https://example.com/b.wav"),
            ],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert [b.block_type for b in blocks] == ["text", "image"]

    def test_enabled_policy_forces_audio_parts_on_unknown_model(self):
        chain = make_chain(
            model_name="some-brand-new-model", media_support={"audio": "enabled"}
        )
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, AudioBlock) for b in blocks)

    def test_undecidable_audio_attaches_fail_loud(self):
        """Unknown model + audio media: attach and let the API error surface."""
        chain = make_chain(model_name="some-brand-new-model")
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, AudioBlock) for b in blocks)

    def test_missing_local_audio_raises(self, tmp_path):
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(
            query="listen", query_media=[("audio", str(tmp_path / "nope.wav"))]
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


class TestMediaRefToAudioBlock:
    """Direct tests of the MediaReference -> AudioBlock mapping."""

    def test_url_ref(self):
        block = media_ref_to_audio_block(
            {
                "content_type": "audio",
                "kind": "url",
                "value": "http://x/a.wav",
                "mime_type": "audio/wav",
            }
        )
        assert isinstance(block, AudioBlock)
        assert str(block.url) == "http://x/a.wav"
        assert block.format is None  # not fetched, so not sniffed

    def test_base64_ref_roundtrips_byte_exact(self):
        raw = WAV_MAGIC
        ref = {
            "content_type": "audio",
            "kind": "base64",
            "value": base64.b64encode(raw).decode(),
            "mime_type": "audio/wav",
        }
        block = media_ref_to_audio_block(ref)
        assert block.format == "wav"
        assert base64.b64decode(block.audio) == raw
        # what the OpenAI adapter path reads must be the original base64 string
        assert block.resolve_audio(as_base64=True).read().decode() == ref["value"]

    @pytest.mark.parametrize("mime_type", sorted(_AUDIO_FORMATS))
    def test_every_supported_audio_mime_has_a_format_token(self, mime_type):
        """The table must cover SUPPORTED_MIME_TYPES['audio'] exactly; a core-side
        addition without a mapping here would KeyError on the request path."""
        from aap_core.utils import SUPPORTED_MIME_TYPES

        assert mime_type in SUPPORTED_MIME_TYPES["audio"]
        block = media_ref_to_audio_block(
            {
                "content_type": "audio",
                "kind": "base64",
                "value": base64.b64encode(b"\x00" * 12).decode(),
                "mime_type": mime_type,
            }
        )
        assert block.format == _AUDIO_FORMATS[mime_type]

    def test_no_unsupported_audio_mime_escapes_the_table(self):
        from aap_core.utils import SUPPORTED_MIME_TYPES

        assert set(SUPPORTED_MIME_TYPES["audio"]) == set(_AUDIO_FORMATS)

    def test_non_audio_ref_rejected(self):
        with pytest.raises(ValueError, match="only audio refs"):
            media_ref_to_audio_block(
                {
                    "content_type": "image",
                    "kind": "base64",
                    "value": "AAAA",
                    "mime_type": "image/png",
                }
            )


class TestMediaRefToVideoBlock:
    """Direct tests of the MediaReference -> VideoBlock mapping."""

    def test_url_ref(self):
        block = media_ref_to_video_block(
            {
                "content_type": "video",
                "kind": "url",
                "value": "http://x/a.mp4",
                "mime_type": "video/mp4",
            }
        )
        assert isinstance(block, VideoBlock)
        assert str(block.url) == "http://x/a.mp4"

    def test_base64_ref_roundtrips_byte_exact(self):
        raw = MP4_MAGIC + b"\x00" * 16
        ref = {
            "content_type": "video",
            "kind": "base64",
            "value": base64.b64encode(raw).decode(),
            "mime_type": "video/mp4",
        }
        block = media_ref_to_video_block(ref)
        assert base64.b64decode(block.video) == raw
        # what an inline-bytes adapter path reads must be the original b64 string
        assert block.resolve_video(as_base64=True).read().decode() == ref["value"]

    @pytest.mark.parametrize("mime_type", sorted(SUPPORTED_MIME_TYPES["video"]))
    def test_video_mime_passes_through_without_a_table(self, mime_type):
        """VideoBlock takes a MIME string, not a format token, so the mapper must
        hand SUPPORTED_MIME_TYPES['video'] entries over unchanged - no table to
        drift out of sync with core."""
        block = media_ref_to_video_block(
            {
                "content_type": "video",
                "kind": "base64",
                "value": base64.b64encode(b"\x00" * 12).decode(),
                "mime_type": mime_type,
            }
        )
        assert block.video_mimetype == mime_type

    def test_fps_is_never_set(self):
        """Frame sampling is provider-side; MediaReference cannot express it."""
        block = media_ref_to_video_block(
            {
                "content_type": "video",
                "kind": "base64",
                "value": base64.b64encode(b"\x00" * 12).decode(),
                "mime_type": "video/mp4",
            }
        )
        assert block.fps is None

    def test_non_video_ref_rejected(self):
        with pytest.raises(ValueError, match="only video refs"):
            media_ref_to_video_block(
                {
                    "content_type": "image",
                    "kind": "base64",
                    "value": "AAAA",
                    "mime_type": "image/png",
                }
            )


class TestOllamaTransportCeiling:
    """The Ollama chat schema has no audio field and its llamaindex adapter raises
    on AudioBlock, so audio must be denied for every Ollama model regardless of
    what the server capabilities list or a model card claims. FakeLLM is not an
    Ollama class, so the ceiling is injected where the chain consults it."""

    def test_ceiling_denies_audio_despite_affirming_server_report(self, monkeypatch):
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"audio": False}
        )
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        assert chain.detect_capabilities()["audio"] is False

    def test_ceiling_denies_audio_despite_affirming_model_card(self, monkeypatch, hf):
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"audio": False}
        )
        hf.results["org/repo"] = {"audio": True}
        chain = make_chain(model_name="org/repo")
        assert chain.detect_capabilities()["audio"] is False

    def test_ollama_audio_media_degrades_to_text(self, monkeypatch, caplog):
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"audio": False}
        )
        chain = make_chain(model_name="qwen2-audio", capabilities=AUDIO_CAPS)
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert len(blocks) == 1
        assert blocks[0].text == "listen"
        assert "does not support audio input" in caplog.text

    def test_denying_audio_does_not_deny_images(self, monkeypatch):
        """The transport carries images; the ceiling must not over-deny."""
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"audio": False}
        )
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, ImageBlock) for b in blocks)

    def test_enabled_policy_overrides_the_ceiling(self, monkeypatch):
        """Explicit declaration short-circuits detection in the core gate, so the
        block is attached and the adapter's own error surfaces. Fail loud."""
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"audio": False}
        )
        chain = make_chain(model_name="qwen2-audio", media_support={"audio": "enabled"})
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, AudioBlock) for b in blocks)

    def test_ollama_transport_still_has_no_audio_field(self):
        """Canary: if this fails, Ollama gained audio and this ceiling must go."""
        assert _ollama_carries("audio") is False


class TestVideoTransportCeiling:
    """No installed llamaindex adapter consumes VideoBlock: the OpenAI converter
    raises on it and the Ollama schema has no field for it, so video must be
    denied on those transports regardless of what a model card claims. The only
    adapter known to deliver video (llama-index-llms-google-genai) is not
    installed here and needs no ceiling: it simply consumes the mapped block."""

    def test_ceiling_denies_video_despite_affirming_model_card(self, monkeypatch, hf):
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"video": False}
        )
        hf.results["org/repo"] = {"video": True}
        chain = make_chain(model_name="org/repo")
        assert chain.detect_capabilities()["video"] is False

    def test_video_denied_by_ceiling_degrades_to_text(self, monkeypatch, caplog):
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"video": False}
        )
        chain = make_chain(model_name="gemma3:12b", capabilities=VISION_CAPS)
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert len(blocks) == 1
        assert blocks[0].text == "watch"
        assert "does not support video input" in caplog.text

    def test_denying_video_does_not_deny_images_or_audio(self, monkeypatch):
        """The transports carry image (and OpenAI carries audio); the ceiling
        must not over-deny the modalities it probed as deliverable."""
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"video": False}
        )
        chain = make_chain(
            model_name="gemma3:12b",
            capabilities=VISION_CAPS,
            media_support={"audio": "enabled"},
        )
        msg = AgentMessage(
            query="look and listen",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("audio", "https://example.com/a.wav"),
                ("video", "https://example.com/a.mp4"),
            ],
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert [b.block_type for b in blocks] == ["text", "image", "audio"]

    def test_enabled_policy_overrides_the_video_ceiling(self, monkeypatch):
        """Explicit declaration short-circuits detection in the core gate, so the
        block is attached and the adapter's own error surfaces. Fail loud."""
        monkeypatch.setattr(
            chain_module, "adapter_capabilities", lambda model: {"video": False}
        )
        chain = make_chain(model_name="gpt-4o", media_support={"video": "enabled"})
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        blocks = user_message_of(chain._prepare_conversation(msg)).blocks
        assert any(isinstance(b, VideoBlock) for b in blocks)

    def test_ollama_transport_still_has_no_video_field(self):
        """Canary: if this fails, Ollama gained video and this ceiling must go."""
        assert _ollama_carries("video") is False

    def test_ollama_transport_carries_images(self):
        """The field scan must affirm the modality the schema does have."""
        assert _ollama_carries("image") is True

    def test_openai_converter_rejects_video_but_carries_image_and_audio(self):
        """Live probe of the installed converter: its ValueError is the verdict.
        If OpenAI gains a video branch, the denial disappears by itself."""
        assert _openai_carries("video") is False
        assert _openai_carries("image") is True
        assert _openai_carries("audio") is True

    def test_dispatch_denies_video_on_openai_transports(self):
        """A model whose class lives in the OpenAI adapter namespace gets the
        converter's verdict without any declaration or detection saying so."""
        assert adapter_capabilities(FakeOpenAILLM()) == {"video": False}

    def test_bedrock_probe_is_undecidable_when_package_absent(self):
        """bedrock-converse is not installed here; an unreadable transport
        reports no denial and leaves the modality to the other sources."""
        assert _bedrock_carries("video") is True

    def test_bedrock_dispatch_denies_what_the_converter_drops(self, monkeypatch):
        """The Bedrock converter warns and returns None for blocks it does not
        map - a silent drop, which is exactly what the ceiling exists to
        prevent. The package is not installed, so a stand-in module supplies
        the dropper's shape: None denies, a dict carries."""

        def fake_convert(block, role):
            return None if block.block_type == "video" else {"text": "ok"}

        monkeypatch.setitem(
            sys.modules, "llama_index.llms.bedrock_converse", SimpleNamespace()
        )
        monkeypatch.setitem(
            sys.modules,
            "llama_index.llms.bedrock_converse.utils",
            SimpleNamespace(_content_block_to_bedrock_format=fake_convert),
        )
        assert adapter_capabilities(FakeBedrockLLM()) == {"video": False}

    def test_unknown_adapters_report_nothing(self):
        """A model outside the checked namespaces stays undecidable: the ceiling
        denies only what it can read from installed code."""
        assert adapter_capabilities(FakeLLM()) == {}


class TestBridgeLayering:
    """The generic-adapter invariant: provider knowledge lives in utils, not chain.

    The frameworks ship hundreds of provider packages; the integration scales
    by keeping the chain a generic pipeline (funnel -> mapper -> gate) and
    confining every provider fact to utils, where it is read from installed
    code rather than hardcoded. Two structural pins keep that true:

    - The mappers are a function of the MediaReference alone: no model
      parameter means no way to branch on model identity.
    - The chain module holds no server probes of its own: the Ollama
      capabilities query moved to utils.server_capabilities.
    """

    @staticmethod
    def _body_source(fn) -> str:
        """The function body as code, docstring removed: prose may name providers."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        definition = tree.body[0]
        definition.body = [
            statement
            for statement in definition.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        return ast.unparse(definition)

    @pytest.mark.parametrize(
        "mapper",
        [media_ref_to_image_block, media_ref_to_audio_block, media_ref_to_video_block],
        ids=["image", "audio", "video"],
    )
    def test_mapper_takes_only_the_reference(self, mapper):
        assert list(inspect.signature(mapper).parameters) == ["ref"]

    @pytest.mark.parametrize(
        "mapper",
        [media_ref_to_image_block, media_ref_to_audio_block, media_ref_to_video_block],
        ids=["image", "audio", "video"],
    )
    def test_mapper_body_names_no_provider_surface(self, mapper):
        body = self._body_source(mapper)
        assert not re.search(r"\bmodel\b|\bprofile\b|__module__|adapter", body)

    def test_chain_holds_no_server_probe(self):
        """Canary for the move: the Ollama query must stay in the utils layer."""
        assert not hasattr(ChatCausalMultiTurnsChain, "_ollama_capabilities")
        source = inspect.getsource(chain_module)
        assert not re.search(r"def _ollama_capabilities|\.show\(", source)
