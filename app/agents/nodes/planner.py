from app.agents.state import AgentState         # Imports the shared state structure used by the agent
from app.gateway import get_langchain_llm  
import logfire  

# Initializes the planner's LLM with fallback, caching, and retry support
llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState):
    """
    Decides whether the user's latest message needs a web search
    or can be answered using the conversation history.
    """

    # Collects all previous messages except the latest user message
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    # Extracts the latest user message
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    # Instructs the LLM to classify the request or generate a search query
    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the message is a greeting or can be answered using conversation history,
       output 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking
       requiring fresh documentation, output a refined search query.

    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    # Tracks the planner's execution and records its decision
    with logfire.span("Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    # Routes conversational requests without requiring retrieval
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    # Passes technical queries to the downstream search/retrieval workflow
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }