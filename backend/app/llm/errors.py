class LLMError(Exception):
    """Raised when a model or embedding provider fails (network, auth, rate limit, bad reply)."""
