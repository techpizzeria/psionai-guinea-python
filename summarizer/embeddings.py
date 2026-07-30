"""Embedding helpers built on the OpenAI Embedding API."""

import openai

EMBEDDING_MODEL = "text-embedding-ada-002"


def embed_text(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Return the embedding vector for ``text``.

    Expects ``openai.api_key`` to already be set by the caller.
    """
    response = openai.Embedding.create(model=model, input=text)
    return response["data"][0]["embedding"]
