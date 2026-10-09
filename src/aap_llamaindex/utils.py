import base64
from typing import Union

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.callbacks.token_counting import get_tokens_from_response
from llama_index.core.llms import ChatResponse, CompletionResponse

from aap_core.types import MediaReference, TokenUsage


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
