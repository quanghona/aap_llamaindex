from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Tuple

from aap_core import utils
from aap_core.chain import BaseCausalMultiTurnsChain
from aap_core.types import AgentMessage, ContentType, TokenUsage
from aap_core.utils import extract_repo_id, huggingface_capabilities
from llama_index.core.base.llms.types import MessageRole, TextBlock
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import BaseTool
from pydantic import Field, PrivateAttr

from aap_llamaindex.utils import media_ref_to_image_block, token_from_response

# Modalities the media gate can consult; detection reports on these.
_GATED_MODALITIES: Tuple[ContentType, ...] = ("image", "audio", "video", "document")


class ChatCausalMultiTurnsChain(BaseCausalMultiTurnsChain[ChatMessage, ChatResponse]):
    model: FunctionCallingLLM = Field(
        ..., description="The LLM model with function calling capability"
    )
    system_prompt: str = Field(..., description="The system prompt")
    user_prompt_template: str = Field(..., description="The user prompt template")

    _tool_dict: Dict[str, BaseTool] = PrivateAttr({})
    _capabilities_cache: Dict[ContentType, bool] | None = PrivateAttr(default=None)
    _capabilities_model: Any = PrivateAttr(default=None)

    def __init__(self, tools: Sequence[BaseTool | Callable] = [], **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def detect_capabilities(self) -> Dict[ContentType, bool]:
        """Resolve per-modality input support from model metadata, not model names.

        Sources, in priority order:
        1. What the user declared in media_support ("enabled"/"disabled"). The
           core gate applies that before this method runs, so an explicit
           declaration always wins.
        2. The Ollama server's capability list for the exact installed revision
           (`ollama show` -> capabilities, e.g. ["completion", "vision"]).
           llama-index-core itself carries no capability metadata - LLMMetadata
           has no capabilities field - so the provider server is asked directly.
        3. Hugging Face model-card metadata, for models identified by repo id
           (HF-native integrations like HuggingFaceLLM or vLLM; needs
           huggingface-hub).

        Model-family names carry no weight here: variants of one family differ
        per modality, so name matching misreports exactly the cases that
        matter. Modalities no source reports on stay absent, and auto mode
        treats them as capable (fail loud).

        The result is cached per model instance and recomputed when the model
        is replaced, since sources 2 and 3 cost network requests.
        """
        if (
            self._capabilities_cache is None
            or self._capabilities_model is not self.model
        ):
            self._capabilities_cache = self._resolve_capabilities()
            self._capabilities_model = self.model
        return dict(self._capabilities_cache)

    def _resolve_capabilities(self) -> Dict[ContentType, bool]:
        """Query the metadata sources once, filling gaps from the next source."""
        capabilities = self._ollama_capabilities()
        if set(capabilities) != set(_GATED_MODALITIES):
            repo_id = extract_repo_id(self._identifier_candidates())
            if repo_id is not None:
                for modality, value in huggingface_capabilities(repo_id).items():
                    capabilities.setdefault(modality, value)
        return capabilities

    def _ollama_capabilities(self) -> Dict[ContentType, bool]:
        """Ask an Ollama-style server what the installed model revision can do.

        Duck-typed on purpose: any integration exposing a `.client` with a
        `.show(name)` returning a `capabilities` list gets consulted (the
        ollama-python client does). A non-empty list is read as complete, so
        absence of "vision" means no image input. Missing client, unknown
        model on the server, or an offline server report undecidable rather
        than breaking the request path.
        """
        name = getattr(self.model, "model", None)
        if not isinstance(name, str):
            return {}
        try:
            show = getattr(self.model.client, "show", None)
            if show is None:
                return {}
            capabilities = show(name).capabilities
        except Exception:  # noqa: BLE001 - detection must never break invoke()
            return {}
        if not isinstance(capabilities, list) or not capabilities:
            return {}
        lowered = [str(capability).lower() for capability in capabilities]
        return {"image": "vision" in lowered}

    def _identifier_candidates(self) -> List[str]:
        """Model identifiers to test for a Hugging Face repo id, best first.

        Plain attributes are read before the framework-native metadata
        property: some integrations (e.g. Ollama) probe their server inside
        `metadata`, so it is consulted only when no attribute already resolves
        to a repo id.
        """
        candidates: List[str] = []
        for holder in (self.model, getattr(self.model, "llm", None)):
            if holder is None:
                continue
            for attr in ("repo_id", "model_id", "model_name", "model"):
                value = getattr(holder, attr, None)
                if isinstance(value, str):
                    candidates.append(value)
        if extract_repo_id(candidates) is not None:
            return candidates
        try:
            name = self.model.metadata.model_name
            if isinstance(name, str) and name != "unknown":
                candidates.append(name)
        except Exception:  # noqa: BLE001 - metadata may hit the network
            pass
        return candidates

    def _prepare_conversation(self, message: AgentMessage) -> List[ChatMessage]:
        user_prompt = self.user_prompt_template.format(**message.format_kwargs())
        images = self._image_parts(message)
        user_content: str | List[Any] = user_prompt
        if images:
            user_content = [
                TextBlock(text=user_prompt),
                *[media_ref_to_image_block(ref) for ref in images],
            ]
        conversation = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]
        total_turns = (
            min(len(message.responses), self.include_history)
            if self.include_history >= 0
            else len(message.responses)
        )
        responses = message.responses[-total_turns:]
        for response in responses:
            if response[0] == "user":
                conversation.append(
                    ChatMessage(role=MessageRole.USER, content=response[1])
                )
            elif response[0] == "tool":
                conversation.append(
                    ChatMessage(role=MessageRole.TOOL, content=response[1])
                )
            elif response[0] == "system":
                conversation.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=response[1])
                )
            else:
                conversation.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=response[1])
                )
        return conversation

    def _generate_response(
        self, conversation: List[ChatMessage], **kwargs
    ) -> Tuple[List[ChatMessage], ChatResponse, bool, TokenUsage]:
        response = self.model.chat_with_tools(
            user_msg=conversation[-1],
            chat_history=conversation[:-1],
            tools=self.tools,
            **kwargs,
        )
        has_tool = False
        for block in response.message.blocks:
            if block.block_type == "text" and len(block.text) > 0:
                block.text = utils.remove_thinking(block.text)
            elif block.block_type == "tool_call":
                has_tool = True
        conversation.append(response.message)
        usage = token_from_response(response)
        return conversation, response, has_tool, usage

    def _process_tools(
        self, conversation: List[ChatMessage], response: ChatResponse
    ) -> List[ChatMessage]:
        tool_calls = self.model.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        # Source: https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/#the-workflow-itself
        # call tools -- safely!
        for tool_call in tool_calls:
            tool = self._tool_dict.get(tool_call.tool_name)
            if tool is None:
                res = f"Tool {tool_call.tool_name} does not exist"
                conversation.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=res,
                        additional_kwargs={},
                    )
                )
            else:
                additional_kwargs = {
                    "tool_call_id": tool_call.tool_id,
                    "name": tool.metadata.get_name(),
                }
                try:
                    tool_output = tool(**tool_call.tool_kwargs)
                    conversation.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=tool_output.content,
                            additional_kwargs=additional_kwargs,
                        )
                    )
                except Exception as e:
                    res = f"Encountered error in tool call: {tool_call.tool_name} {e}"
                    conversation.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=res,
                            additional_kwargs=additional_kwargs,
                        )
                    )

        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[ChatMessage]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 2
            if self.store_immediate_steps
            else len(conversation) - 1
        )  # 2 is system message and user query
        end_index = len(conversation)
        name_map = {
            MessageRole.ASSISTANT: self.name,
            MessageRole.CHATBOT: self.name,
            MessageRole.FUNCTION: "tool",
            MessageRole.DEVELOPER: "system",
            MessageRole.MODEL: self.name,
            MessageRole.USER: "user",
            MessageRole.TOOL: "tool",
            MessageRole.SYSTEM: "system",
        }
        for i in range(start_index, end_index):
            # Extract text from all TextBlocks in the message
            text_parts = []
            for block in conversation[i].blocks:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
            if text_parts:
                message.responses.append(
                    (name_map[conversation[i].role], "\n".join(text_parts))  # type: ignore
                )
            # TODO: handle other modals later

        return message

    @property
    def tools(self) -> List[BaseTool]:
        return list(self._tool_dict.values())

    @tools.setter
    def tools(self, tools: Sequence[BaseTool | Callable]):
        if len(tools) > 0 and isinstance(tools[0], Callable):
            tools = [FunctionTool.from_defaults(tool) for tool in tools]
        self._tool_dict = {tool.metadata.get_name(): tool for tool in tools}  # type: ignore
