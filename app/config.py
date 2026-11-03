import os
from dotenv import load_dotenv

load_dotenv()

class settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")


    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "enterprise_rag"

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "rag"     
    GROQ_SLUG_2 = "rag2"

    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")    

settings = settings()
