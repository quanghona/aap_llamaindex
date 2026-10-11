import base64
from typing import Any, Dict, Union

from llama_index.core.base.llms.types import AudioBlock, ImageBlock
from llama_index.core.callbacks.token_counting import get_tokens_from_response
from llama_index.core.llms import ChatResponse, CompletionResponse

from aap_core.types import ContentType, MediaReference, TokenUsage


def token_from_response(usage: Union[CompletionResponse, ChatResponse]) -> TokenUsage:
    """
    Convert llamaindex's response object to a TokenUsage object.

    Args:
        usage (Union[CompletionResponse, ChatResponse]): The response object to convert.

    Returns:
        TokenUsage: The converted object.
    """
    input_tokens, output_tokens = get_tokens_from_response(usage)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def media_ref_to_image_block(ref: MediaReference) -> ImageBlock:
    """Convert a resolved MediaReference to a llama-index ImageBlock.

    base64 refs are decoded to raw bytes; ImageBlock re-encodes and stores them
    internally, so passing the core-sniffed mime_type explicitly preserves the
    detection for formats the block's own sniffing may miss. URL refs pass
    through; integrations fetch them via ImageBlock.resolve_image when the
    provider API needs inline bytes (e.g. llama-index-llms-ollama).

    Other modalities (audio, video, document) have their own block classes in
    llama-index-core and get their own mappers when wired up; this helper
    refuses non-image refs rather than silently mislabeling them.

    Args:
        ref (MediaReference): The resolved reference from BaseLLMChain._media_parts.

    Returns:
        ImageBlock: A native block ready to embed in a ChatMessage's blocks.
    """
    if ref["content_type"] != "image":
        raise ValueError(
            f"media_ref_to_image_block got content_type={ref['content_type']!r}; "
            "only image refs are mapped today."
        )
    if ref["kind"] == "url":
        return ImageBlock(url=ref["value"])
    return ImageBlock(
        image=base64.b64decode(ref["value"]), image_mimetype=ref["mime_type"]
    )


# MediaReference.mime_type -> AudioBlock.format. AudioBlock stores a filetype-
# style extension token, not a MIME string, and consumers read it as such:
# llama-index-llms-openai passes it straight into OpenAI's input_audio.format,
# whose enum accepts only "wav" and "mp3". Deriving the token from the MIME
# subtype (as langchain's translator does) yields "mpeg" and is rejected, so
# the mapping is pinned here over the full SUPPORTED_MIME_TYPES["audio"] set.
# Where filetype would guess differently it is wrong for our purposes: it
# reports "ogg" for Opus streams, nothing for amr/webm, and "wmv" for WMA.
_AUDIO_FORMATS: Dict[str, str] = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/aiff": "aiff",
    "audio/amr": "amr",
    "audio/webm": "webm",
    "audio/x-ms-wma": "wma",
}


def media_ref_to_audio_block(ref: MediaReference) -> AudioBlock:
    """Convert a resolved MediaReference to a llama-index AudioBlock.

    base64 refs are handed over still base64-encoded: AudioBlock's validator
    keeps a payload that decodes cleanly as-is and only encodes raw bytes, so
    passing the core's base64 string avoids the validator's ambiguous path
    (arbitrary bytes can "decode" as base64 when invalid characters are
    discarded). The explicit format token suppresses the block's own filetype
    sniffing, which is weaker than the core's container probes.

    URL refs pass through without a format token; fetching a remote file to
    sniff it is the provider integration's job (resolve_audio), not ours.

    Args:
        ref (MediaReference): The resolved reference from BaseLLMChain._media_parts.

    Returns:
        AudioBlock: A native block ready to embed in a ChatMessage's blocks.
    """
    if ref["content_type"] != "audio":
        raise ValueError(
            f"media_ref_to_audio_block got content_type={ref['content_type']!r}; "
            "only audio refs are mapped here."
        )
    if ref["kind"] == "url":
        return AudioBlock(url=ref["value"])
    return AudioBlock(
        audio=ref["value"].encode("utf-8"), format=_AUDIO_FORMATS[ref["mime_type"]]
    )


def _ollama_carries_audio() -> bool:
    """Report whether the Ollama chat request schema can carry an audio part.

    This is a fact about the wire format, not about a model family: the message
    type has one field per media kind (today images, nothing for audio), so if
    no audio field exists, no Ollama model can receive audio regardless of what
    a server capabilities list or model card claims. Read from the installed
    schema rather than hardcoded, so the answer updates when the transport
    gains one. An unreadable schema reports True, which leaves the modality
    undecidable and keeps auto mode fail-loud.
    """
    try:
        from ollama._types import Message

        fields = set(Message.model_fields)
    except Exception:  # noqa: BLE001 - optional dependency, private module path
        return True
    return any("audio" in field.lower() for field in fields)


def adapter_capabilities(model: Any) -> Dict[ContentType, bool]:
    """Map an adapter's transport onto the modalities it can physically deliver.

    Complements the server/model-card capability sources: those state what a
    model understands, this states what the request format can carry, and the
    second is a hard ceiling on the first. Only denials are returned - a
    transport that names an audio field proves nothing about a given model
    accepting it.

    The module-path test is a prefix match on the llamaindex Ollama namespace:
    unlike langchain, every llamaindex integration lives under llama_index.*,
    so a top-level-package check would match all models, not just Ollama's.

    Args:
        model: A LlamaIndex LLM instance.

    Returns:
        Dict[ContentType, bool]: Modalities the transport cannot deliver, as False.
    """
    for klass in type(model).__mro__:
        if klass.__module__.startswith("llama_index.llms.ollama"):
            if not _ollama_carries_audio():
                return {"audio": False}
            return {}
    return {}
