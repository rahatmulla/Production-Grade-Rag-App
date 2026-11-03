# text embeddings are generated using Gemini
# switches to a local model(sentence-transformers) if Gemini is unavailable,
# processes texts in batches with retries for rate limits.


import time 
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings                                     #to import the variables created in the config file

BATCH_SIZE = 50
_GEMINI_DIM = 3072                                                  #MORE DIMENSION ->MORE SEMANTIC SPACE -> MORE ACCURATE RESULTS , but models give fixed dimensioms  
_FALLBACK_DIM = 768                                    # all-mpnet-base-v2 , we use this model when we hit the rate limits

_active_model = None
_model_type : str | None = None       #gemini or fallback


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")                                                              #embedding the text "probe" to check if the model is reachable
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None

def _load_fallback():                                                                               #falls back to sentence transformer if gemini not available
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")

def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"                

# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Exponential backoff: 1 s → 2 s → 4 s → 8 s (4 attempts total)
        for attempt in range(4):                                                 # Retry up to 4 times if a rate limit occurs
            try:
                return _active_model.embed_documents(batch)                                         # Convert the batch of text into embedding vectors
            except Exception as e:
                err = str(e).lower()                                                 # Convert the error message to lowercase for easier checking
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))       # Check whether the error is related to rate limits or quota
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt                                                 # Wait 1, 2, then 4 seconds before retrying
                    logfire.warning(                                                 # Log a warning with the retry delay and attempt number
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")              # Raise an error if all four attempts fail due to rate limits
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()            # For other models, generate embeddings and convert the result to a list


# Generate an embedding for a single query
def embed_query(query: str) -> list[float]:
    # Initialize the embedding model
    _init()

    # Use Gemini or the selected model to generate the embedding
    if _model_type == "gemini":
        return _active_model.embed_query(query)

    # Convert the query into an embedding vector and return it as a list
    return _active_model.encode([query])[0].tolist()


# Generate embeddings for multiple texts
def embed_texts(texts: list[str]) -> list[list[float]]:
    # Initialize the embedding model
    _init()

    # Store all generated embeddings
    all_embeddings: list[list[float]] = []

    # Process texts in smaller batches
    for i in range(0, len(texts), BATCH_SIZE):
        # Select the current batch of texts
        batch = texts[i : i + BATCH_SIZE]

        # Track the embedding process using Logfire
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            # Generate embeddings for the batch and add them to the results
            all_embeddings.extend(_embed_batch(batch))

    # Return embeddings for all texts
    return all_embeddings
