import os
from enum import Enum
from typing import Any, Optional

from chromadb import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    ONNXMiniLM_L6_V2,
    OpenAIEmbeddingFunction,
    CohereEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)


class SupportedEmbeddingFunctions(str, Enum):
    default = "default"
    openai = "openai"
    cohere = "cohere"
    hf = "hf"
    st = "st"


def get_embedding_function_for_name(
    name: Optional[SupportedEmbeddingFunctions], **kwargs: Any
) -> EmbeddingFunction:
    if name == SupportedEmbeddingFunctions.default:
        return ONNXMiniLM_L6_V2()
    elif name == SupportedEmbeddingFunctions.openai:
        model = (
            kwargs.get("model")
            if kwargs.get("model")
            else os.environ.get("OPENAI_MODEL_NAME", "text-embedding-ada-002")
        )
        return OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"), model_name=model
        )
    elif name == SupportedEmbeddingFunctions.cohere:
        model = (
            kwargs.get("model")
            if kwargs.get("model")
            else os.environ.get("COHERE_MODEL_NAME", "embed-english-v3.0")
        )
        return CohereEmbeddingFunction(
            api_key=os.environ.get("COHERE_API_KEY"), model_name=model
        )
    elif name == SupportedEmbeddingFunctions.hf:
        model = (
            kwargs.get("model")
            if kwargs.get("model")
            else os.environ.get(
                "HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        return HuggingFaceEmbeddingFunction(
            api_key=os.environ.get("HF_TOKEN"), model_name=model
        )
    elif name == SupportedEmbeddingFunctions.st:
        model = (
            kwargs.get("model")
            if kwargs.get("model")
            else os.environ.get("ST_MODEL_NAME", "all-MiniLM-L6-v2")
        )
        return SentenceTransformerEmbeddingFunction(
            model_name=model,
            device=os.environ.get("ST_DEVICE", "cpu"),
            normalize_embeddings=os.environ.get("ST_NORMALIZE", "True") == "True",
        )
    else:
        raise ValueError("Please provide a valid embedding function.")
