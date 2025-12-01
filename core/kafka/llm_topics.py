from enum import Enum

class LlmKafkaTopic(str, Enum):
    CHAT_REQUEST = "llm.chat.request"
    CHAT_RESPONSE = "llm.chat.response"
    CHAT_TOKEN = "llm.chat.token"