
# Imports required libraries for monitoring, Portkey gateway integration, LangChain, and application settings
import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Configures the LLM gateway with fallback models, response caching, and retries for rate limits and server errors
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {
            "override_params": {
                "model": f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"
            }
        },
        {
            "override_params": {
                "model": f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b"
            }
        }
    ]
}

# Initializes the Portkey client with API authentication and gateway configuration
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG_SLUG,
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Creates a LangChain-compatible LLM client that sends requests through Portkey.
    Configures the primary Groq model, request settings, gateway routing, and metadata.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_SLUG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )


def extract_cache_status(response) -> str:
    """
    Extracts the cache status from the Portkey response headers.
    Checks multiple possible response attributes and returns 'MISS' if unavailable.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
