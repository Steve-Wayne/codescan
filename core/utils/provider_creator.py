"""
This module provides util methods used for initializing an AIProvider based on the user args.
"""

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-pro",
}


def _get_provider_class(provider):
    if provider == "openai":
        from core.providers.open_ai_provider import OpenAIProvider

        return OpenAIProvider
    if provider == "gemini":
        from core.providers.google_gemini_ai_provider import GoogleGeminiAIProvider

        return GoogleGeminiAIProvider
    if provider == "custom":
        from core.providers.custom_ai_provider import CustomAIProvider

        return CustomAIProvider
    raise ValueError(f"Unsupported provider: {provider}")


def init_provider(provider, model, host=None, port=None, token=None, endpoint=None): # pylint: disable=R0917
    """
    Initializes and returns the appropriate AI client based on the provider.
    """

    if provider == "custom":
        client_params = {
            "model": model,
            "host": host,
            "port": port,
            "token": token,
            "endpoint": endpoint,
        }
    else:
        client_params = {
            "model": model if model else DEFAULT_MODELS[provider],
        }

    provider_class = _get_provider_class(provider)
    return provider_class(**client_params)
