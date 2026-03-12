"""Embedding model wrapper for the RAG pipeline."""

from langchain_openai import OpenAIEmbeddings


def build_embedder(
    model: str = "text-embedding-ada-002",
    api_key: str | None = None,
) -> OpenAIEmbeddings:
    """Create an :class:`~langchain_openai.OpenAIEmbeddings` instance.

    Args:
        model: Name of the OpenAI embedding model to use.
        api_key: OpenAI API key. If ``None``, the value is read from the
            ``OPENAI_API_KEY`` environment variable.

    Returns:
        A configured :class:`~langchain_openai.OpenAIEmbeddings` object.
    """
    kwargs: dict = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)
