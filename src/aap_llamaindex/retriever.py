from aap_core.retriever import BaseRetriever
from aap_core.types import AgentMessage
from llama_index.core.base.base_retriever import (
    BaseRetriever as LLamaIndexBaseRetriever,
)
from pydantic import ConfigDict, Field, field_validator


class RetrieverAdapter(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    retriever: LLamaIndexBaseRetriever = Field(
        ..., description="The llamaindex's retriever to use"
    )
    data_key: str = Field(
        default="context.data", description="The key to the data in the message"
    )

    @field_validator("data_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context."):
            raise ValueError("data_key must start with 'context.'")
        return v

    def retrieve(self, message: AgentMessage, **kwargs) -> AgentMessage:
        nodes = self.retriever.retrieve(message.query)
        data = []
        # TODO: aware for multimodal data in the future
        for node in nodes:
            data.append(node.get_text())

        data_key = self.data_key.replace("context.", "")
        if message.context is None:
            message.context = {data_key: data}
        else:
            message.context[data_key] = data
        return message
