import os
from enum import Enum

from chromadb import EmbeddingFunction
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2, OpenAIEmbeddingFunction


class SupportedEmbeddingFunctions(str, Enum):
    default = "default"
    openai = "openai"


def get_embedding_function_for_name(
    name: SupportedEmbeddingFunctions,
) -> EmbeddingFunction:
    if name == SupportedEmbeddingFunctions.default:
        return ONNXMiniLM_L6_V2()
    elif name == SupportedEmbeddingFunctions.openai:
        return OpenAIEmbeddingFunction(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        raise ValueError("Please provide a valid embedding function.")
