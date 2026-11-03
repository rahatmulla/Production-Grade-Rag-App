#Creates a FastAPI backend for a RAG application that validates questions, blocks out-of-scope queries, processes valid questions, and monitors errors.
# ============================================================
# CRITICAL: Configure Logfire before importing application modules.
# ============================================================

#Load environment variables and configure Logfire
import os

import logfire
from dotenv import load_dotenv

load_dotenv()
if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
else:
    logfire.configure(send_to_logfire=False)

# Import application modules after Logfire configuration.
from fastapi import FastAPI , Request
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from fastapi.responses import Response

from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)      #Configure rate limiting, no of requests sent permin from one ip address

#Initialize guardrails at application startup
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize guardrails when the API starts."""
    initialize_rails()
    yield

#Create the FastAPI application
app = FastAPI(
    title="Enterprise Agentic RAG API",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

logfire.instrument_fastapi(app)

class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=5000)
    thread_id: Optional[str] = Field(
        default="default_user",
        min_length=1,
        max_length=100,
    )

    #Validate incoming questions and thread IDs
    @field_validator("q")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query cannot be empty or whitespace only.")

        return value

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("Thread ID cannot be empty or whitespace only.")

        return value.strip() if value is not None else value

#Create the home endpoint
@app.get("/")
def home():
    """Confirm that the API is running."""
    return {"message": "Enterprise LangGraph RAG API is live."}

#Create the graph visualization endpoint#
@app.get("/graph")
def get_graph_image():
    """Return a PNG image of the agent workflow."""
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception:
        logfire.exception("Could not generate graph image.")
        return {"error": "Could not generate graph image."}

#Create the query endpoint with rate limiting
@app.post("/query")
@limiter.limit("10/minute")                                         #user can send a max of 10 requests per min from the same IP address to this endpoint
def query(request: Request, payload: QueryRequest):
    """Check query scope before running the RAG workflow."""
    q = payload.q                                          #Prepare the query and thread ID 
    thread_id = payload.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: Block out-of-scope queries before retrieval.
        blocked, block_response = guard(q)

        if blocked:
            logfire.info(
                f"Request blocked by scope check | thread={thread_id}"
            )
            return {
                "question": q,
                "answer": block_response,
                "thought_process": [
                    "Scope check: Blocked",
                    "Retrieval: Skipped",
                ],
                "status": "Blocked by scope check.",
                "sources": [],
            }

        # Gate 2: Run the LangGraph RAG workflow.
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
        }

    except Exception:
        logfire.exception(
            f"Backend execution failed | thread={thread_id}"
        )
        return {
            "question": q,
            "answer": (
                "I encountered an internal error while processing "
                "your request. Please try again later."
            ),
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }
