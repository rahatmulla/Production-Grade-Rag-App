#This code generates final AI response using either conversation history or RAG, depending on the planner's decision.
# It uses Portkey to call the LLM, checks whether the response was cached, logs execution with Logfire, and updates the agent's state.

import logfire
from app.agents.state import AgentState                         #imports shared state managed by agent
from app.gateway import portkey_client, extract_cache_status  # Imports the Portkey client and cache-status helper


RESPONSE_STYLE = (
    "Lead with the direct answer. Aim for 2–4 short sentences by default. "
    "Use a brief numbered list for procedures. Expand when the user asks for "
    "more detail or when essential steps or caveats require it. "
    "Avoid generic introductions and repeated summaries."
)


def generate_node(state: AgentState):
    """
    Generates the final response using conversation history or retrieved documents.
    Uses Portkey to call the LLM and check whether the response came from cache.
    """

    # Gets the current query or intent identified by the planner
    query = state["current_query"]

    # Builds a readable conversation history, excluding the latest message
    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    # Extracts the latest user message
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    # Uses conversation history only when no document retrieval is required
    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")

        # Creates a prompt for answering conversational questions
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.
        {RESPONSE_STYLE}

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else:
        logfire.info("Generating technical RAG response.")

        # Limits the retrieved context size to control token usage
        max_context_chars = 25000
        full_context = ""

        # Combines retrieved documents until the character limit is reached
        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                # Records when additional document context is excluded
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        # Creates a prompt using technical context and conversation history
        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.
        {RESPONSE_STYLE}

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    # Tracks the response-generation process
    with logfire.span("LLM Synthesis"):
        try:
            # Sends the prompt to the LLM through the Portkey gateway
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # Extracts the generated answer from the API response
            content = response.choices[0].message.content

            # Checks whether Portkey served the response from its cache
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                # Updates the execution plan and status when a cached response is used
                logfire.info("Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit"]
                status = "Cache hit — instant response."
            else:
                # Records normal response generation when no cache hit is detected
                logfire.info("Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            # Returns the generated answer and updates the agent's shared state
            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            # Logs the error and propagates it to the caller
            logfire.error(f"LLM Generation failed: {e}")
            raise