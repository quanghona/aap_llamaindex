import base64
from typing import Any, Dict, Union

from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ImageBlock,
    MessageRole,
    VideoBlock,
)
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


def media_ref_to_video_block(ref: MediaReference) -> VideoBlock:
    """Convert a resolved MediaReference to a llama-index VideoBlock.

    base64 refs are handed over still base64-encoded, exactly as for audio:
    VideoBlock's validator keeps a payload that decodes cleanly as-is, so the
    core's base64 string passes through without a re-encode round-trip. The
    core-sniffed mime_type is set explicitly as video_mimetype because the
    block's own filetype sniffing is weaker than the core's container probes.

    Unlike AudioBlock, VideoBlock carries a MIME string, not a filetype-style
    format token, so no mapping table is needed: SUPPORTED_MIME_TYPES["video"]
    passes through unchanged. The fps field is deliberately left unset - frame
    sampling is the provider's decision (Gemini samples at 1 FPS by default),
    and MediaReference cannot express a sampling policy. Video is a container
    here, never a frame sequence.

    URL refs pass through without a mimetype; fetching a remote file to sniff
    it is the provider integration's job (resolve_video), not ours.

    Args:
        ref (MediaReference): The resolved reference from BaseLLMChain._media_parts.

    Returns:
        VideoBlock: A native block ready to embed in a ChatMessage's blocks.
    """
    if ref["content_type"] != "video":
        raise ValueError(
            f"media_ref_to_video_block got content_type={ref['content_type']!r}; "
            "only video refs are mapped here."
        )
    if ref["kind"] == "url":
        return VideoBlock(url=ref["value"])
    return VideoBlock(
        video=ref["value"].encode("utf-8"), video_mimetype=ref["mime_type"]
    )


# Modalities the transport ceilings in adapter_capabilities can form an opinion
# on. Document is excluded on purpose: DocumentBlock requires a filename
# MediaReference does not carry, so a probe would confuse a shape mismatch with
# a transport refusal (and would break the never-deny-document rule).
_CEILING_MODALITIES = ("image", "audio", "video")


def _probe_block(modality: ContentType) -> Any:
    """Build a minimal well-formed block of one modality for transport probes.

    The payloads are valid base64 (three NUL bytes) so the block validators
    keep them as-is; probes care only about whether the converter accepts the
    block type, never about the media content.
    """
    if modality == "image":
        return ImageBlock(image=b"AAAA")
    elif modality == "audio":
        return AudioBlock(audio=b"AAAA", format="wav")
    elif modality == "video":
        return VideoBlock(video=b"AAAA")
    return None  # document and unknown modalities are never probed


def _ollama_carries(modality: ContentType) -> bool:
    """Report whether the Ollama chat request schema can carry a modality part.

    This is a fact about the wire format, not about a model family: the message
    type has one field per media kind (today images, nothing for audio or
    video), so if no field names one, no Ollama model can receive it regardless
    of what a server capabilities list or model card claims. Read from the
    installed schema rather than hardcoded, so the answer updates when the
    transport gains a field. An unreadable schema reports True, which leaves
    the modality undecidable and keeps auto mode fail-loud.
    """
    try:
        from ollama._types import Message

        fields = set(Message.model_fields)
    except Exception:  # noqa: BLE001 - optional dependency, private module path
        return True
    return any(modality in field.lower() for field in fields)


def _openai_carries(modality: ContentType) -> bool:
    """Report whether the OpenAI converter can emit a modality block.

    Every OpenAI-protocol llamaindex adapter builds its request bodies with
    to_openai_message_dict, so a block type it refuses is undeliverable through
    that transport whatever a server list or model card claims. The probe is a
    well-formed base64 block of the modality, matching what the media_ref_to_*
    mappers produce: a ValueError is the converter's own verdict ("Unsupported
    content block type: VideoBlock"), while an ImportError or any other failure
    means the probe itself broke, which stays undecidable and keeps auto mode
    fail-loud. Read from the installed code rather than hardcoded, so the
    answer updates when the converter gains a block type. Only the
    chat-completions converter is probed; the responses-API converter rejects
    the same block types, so the verdict is transport-wide.
    """
    try:
        from llama_index.llms.openai.utils import to_openai_message_dict
    except ImportError:
        return True
    try:
        message = ChatMessage(role=MessageRole.USER, content=[_probe_block(modality)])
    except Exception:  # noqa: BLE001 - a broken probe must not deny anything
        return True
    try:
        to_openai_message_dict(message)
    except ValueError:
        return False
    except Exception:  # noqa: BLE001 - a broken probe must not deny anything
        return True
    return True


def _bedrock_carries(modality: ContentType) -> bool:
    """Report whether the Bedrock Converse converter can emit a modality block.

    Unlike the raising converters, this one silently drops blocks it does not
    know (logger.warning + None), which is exactly the failure the gate exists
    to prevent - so the probe reads the dropper's own verdict: a None result
    denies the modality before the block can vanish from the request. The
    dispatch only reaches this check when the package is installed, and its
    converter imports its dependencies at module level, so an ImportError here
    means the transport could not serve any request anyway; report True and
    leave the modality undecidable.
    """
    try:
        from llama_index.llms.bedrock_converse.utils import (
            _content_block_to_bedrock_format,
        )
    except ImportError:
        return True
    try:
        carried = _content_block_to_bedrock_format(
            _probe_block(modality), MessageRole.USER
        )
    except Exception:  # noqa: BLE001 - a broken probe must not deny anything
        return True
    return carried is not None


def server_capabilities(model: Any) -> Dict[ContentType, bool]:
    """Ask an Ollama-style server what the installed model revision can do.

    Provider knowledge lives in this module, never in the chain: llama-index-core
    itself carries no capability metadata - LLMMetadata has no capabilities
    field - so the only model-level evidence for most HF-native integrations is
    the model card (consulted by the chain through aap_core), and for Ollama it
    is the server's own list. Duck-typed on purpose: any integration exposing a
    `.client` with a `.show(name)` returning a `capabilities` list gets
    consulted (the ollama-python client does). A non-empty list is read as
    complete, so absence of "vision" means no image input and absence of
    "audio" means no audio input. Missing client, unknown model on the server,
    or an offline server report undecidable rather than breaking the request
    path.

    Args:
        model: A LlamaIndex LLM instance.

    Returns:
        Dict[ContentType, bool]: Modalities the server reported on, as booleans.
    """
    name = getattr(model, "model", None)
    if not isinstance(name, str):
        return {}
    try:
        show = getattr(model.client, "show", None)
        if show is None:
            return {}
        capabilities = show(name).capabilities
    except Exception:  # noqa: BLE001 - detection must never break invoke()
        return {}
    if not isinstance(capabilities, list) or not capabilities:
        return {}
    lowered = [str(capability).lower() for capability in capabilities]
    return {"image": "vision" in lowered, "audio": "audio" in lowered}


# Transport checkers keyed on the module prefix of the adapters they describe.
# The OpenAI-protocol entries share one converter: azure_openai, openai_like
# and together all build requests through llama_index.llms.openai's
# to_openai_message_dict (together subclasses OpenAILike, which subclasses
# OpenAI), so probing the converter is correct for each of them.
_CEILING_CHECKS = (
    ("llama_index.llms.ollama", _ollama_carries),
    ("llama_index.llms.openai", _openai_carries),
    ("llama_index.llms.azure_openai", _openai_carries),
    ("llama_index.llms.openai_like", _openai_carries),
    ("llama_index.llms.together", _openai_carries),
    ("llama_index.llms.bedrock_converse", _bedrock_carries),
)


def adapter_capabilities(model: Any) -> Dict[ContentType, bool]:
    """Map an adapter's transport onto the modalities it can physically deliver.

    Complements the server/model-card capability sources: those state what a
    model understands, this states what the request format can carry, and the
    second is a hard ceiling on the first. Only denials are returned - a
    transport that names an image field, or converts an audio block, proves
    nothing about a given model accepting that media.

    The module-path test is a prefix match on the adapter namespaces: unlike
    langchain, every llamaindex integration lives under llama_index.*, so a
    top-level-package check would match all models, not just the intended
    ones. Adapters outside _CEILING_CHECKS report nothing, leaving every
    modality undecidable.

    Args:
        model: A LlamaIndex LLM instance.

    Returns:
        Dict[ContentType, bool]: Modalities the transport cannot deliver, as False.
    """
    for klass in type(model).__mro__:
        for prefix, carries in _CEILING_CHECKS:
            if klass.__module__.startswith(prefix):
                return {
                    modality: False
                    for modality in _CEILING_MODALITIES
                    if not carries(modality)
                }
    return {}
