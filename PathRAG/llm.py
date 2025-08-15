import os
import numpy as np
from dotenv import load_dotenv
import litellm
from .utils import wrap_embedding_func_with_attrs, logger
from typing import Union

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
# LiteLLM will automatically look for provider-specific API keys
# like GEMINI_API_KEY, OPENAI_API_KEY, etc. in the environment.
# So, we just need to ensure they are loaded.

async def litellm_completion(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs,
) -> Union[str,]:
    """
    Generates a response from the LLM using LiteLLM.
    """
    model = os.getenv("LITELLM_MODEL", "gemini/gemini-1.5-flash-latest")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")

    # Remove arguments that are not part of litellm's standard completion call
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content

# Default to a Gemini embedding model, as requested by the user for the main LLM.
# The user can override this by setting LITELLM_EMBEDDING_MODEL in the .env file.
# Gemini's `embedding-001` model has 768 dimensions.
@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def litellm_embedding(
    texts: list[str],
    **kwargs,
) -> np.ndarray:
    """
    Generates embeddings using LiteLLM.
    """
    # The model is specified in the .env file or defaults to embedding-001
    model = os.getenv("LITELLM_EMBEDDING_MODEL", "embedding-001")

    # LiteLLM's aembedding function is a coroutine
    response = await litellm.aembedding(
        model=model,
        input=texts,
        **kwargs,
    )

    # The response is a ModelResponse object, we need to access the `data` attribute which is a list of dicts.
    return np.array([item.embedding for item in response.data])

