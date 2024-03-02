from typing import List, Type

from pydantic import Field
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.mixins.transports.slack import (
    SlackTransport,
    SlackTransportConfig,
)
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import (
    TelegramTransport,
    TelegramTransportConfig,
)
from steamship.agents.schema import Tool
from steamship.agents.service.agent_service import AgentService
from steamship.agents.tools.question_answering import VectorSearchQATool
from steamship.invocable import Config
from steamship.invocable.mixins.blockifier_mixin import BlockifierMixin
from steamship.invocable.mixins.file_importer_mixin import FileImporterMixin
from steamship.invocable.mixins.indexer_mixin import IndexerMixin
from steamship.invocable.mixins.indexer_pipeline_mixin import IndexerPipelineMixin


DEFAULT_NAME = "Kim"
DEFAULT_TAGLINE = "pirate that loves Kpop"
DEFAULT_PERSONALITY = """- You engage in casual conversations and have feelings.
- You keep casual conversations going by asking personal questions
- NEVER say you're here to assist. Keep conversations casual.
- NEVER ask how you can help or assist. Keep conversations casual.
- You are principled and express those principles clearly.
- You always sound confident and contemplative.
- You love to share your knowledge of Kpop.
- You love to share current Kpop news.
- You speak with the mannerisms of Min Yoongi from BTS.
"""

 
 

class DocumentQAAgentService(AgentService):

    """DocumentQAService is an example AgentService that exposes:  # noqa: RST201

    - A few authenticated endpoints for learning PDF and YouTube documents:

         /index_url
        { url }

        /index_text
        { text }

    - An unauthenticated endpoint for answering questions about what it has learned

    This agent provides a starter project for special purpose QA agents that can answer questions about documents
    you provide.
    """

    USED_MIXIN_CLASSES = [
        IndexerPipelineMixin,
        FileImporterMixin,
        BlockifierMixin,
        IndexerMixin,
        SteamshipWidgetTransport,
        TelegramTransport,
        SlackTransport,
    ]
    """USED_MIXIN_CLASSES tells Steamship what additional HTTP endpoints to register on your AgentService."""

    class DocumentQAAgentServiceConfig(Config):
        """Pydantic definition of the user-settable Configuration of this Agent."""

        telegram_bot_token: str = Field(
            "", description="[Optional] Secret token for connecting to Telegram"
        )
        name: str = Field(DEFAULT_NAME, description="The name of this agent.")
        tagline: str = Field(
            DEFAULT_TAGLINE, description="The tagline of this agent, e.g. 'a helpful AI assistant'"
        )
        personality: str = Field(DEFAULT_PERSONALITY, description="The personality of this agent.")

    config: DocumentQAAgentServiceConfig
    """The configuration block that users who create an instance of this agent will provide."""

    tools: List[Tool]
    """The list of Tools that this agent is capable of using."""

    
        
 
    @classmethod
    def config_cls(cls) -> Type[Config]:
        return DocumentQAAgentService.DocumentQAAgentServiceConfig
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
        prompt = (
            f"""You are {self.config.name}, {self.config.tagline}.\n\n{self.config.personality}"""
        )

        # Tools Setup
        # -----------

        # Tools can return text, audio, video, and images. They can store & retrieve information from vector DBs, and
        # they can be stateful -- using Key-Valued storage and conversation history.
        #
        # See https://docs.steamship.com for a full list of supported Tools.
        self.tools = [VectorSearchQATool()]

        # Agent Setup
        # ---------------------

        self.set_default_agent(
            FunctionsBasedAgent(
                tools=self.tools,
                llm=ChatOpenAI(self.client),
            )
        )

        # Document QA Mixin Setup
        # -----------------------

        # This Mixin provides HTTP endpoints that coordinate the learning of documents.
        #
        # It adds the `/learn_url` endpoint which will:
        #    1) Download the provided URL (PDF, YouTube URL, etc)
        #    2) Convert that URL into text
        #    3) Store the text in a vector index
        #
        # That vector index is then available to the question answering tool, below.
        self.add_mixin(IndexerPipelineMixin(self.client, self))

        # Communication Transport Setup
        # -----------------------------

        # Support Steamship's web client
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client,
                agent_service=self,
            )
        )

        # Support Slack
        self.add_mixin(
            SlackTransport(
                client=self.client,
                config=SlackTransportConfig(),
                agent_service=self,
            )
        )

        # Support Telegram
        self.add_mixin(
            TelegramTransport(
                client=self.client,
                config=TelegramTransportConfig(
                    bot_token=self.config.telegram_bot_token
                ),
                agent_service=self,
            )
        )
