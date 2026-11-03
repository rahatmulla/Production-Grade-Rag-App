# Graph allows all 3 nodes to connect.

from langgraph.graph import StateGraph, END                                     # stategraph allows state to be passed between nodes, END is used to terminate the graph
from langgraph.checkpoint.memory import MemorySaver                             #local memory saver to store conversation history
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node                              #import the planner node
from app.agents.nodes.retriever import retrieve_node                            
from app.agents.nodes.responder import generate_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)                                  
workflow.add_node("retriever", retrieve_node)                            # Add retriever node
workflow.add_node("responder", generate_node)

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == "CONVERSATIONAL":                      #if query is conversational, route to responder
        return "responder"
    return "retriever"                                                  #or else route to retriever

workflow.set_entry_point("planner")                                        #entry point of the graph is the planner node   


# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(                                                 #conditional edges- dynamic routing based on the planner's output
    "planner",
    route_planner,
    {
        "retriever": "retriever",                                                   
        "responder": "responder"
    }
)


workflow.add_edge("retriever", "responder")                                     #retriever node ->responder node
workflow.add_edge("responder", END)                                              #responder node -> END node (terminates the graph)


# --- MEMORY UPGRADE ---
# MemorySaver allows the agent to remember conversations based on 'thread_id'
checkpointer = MemorySaver()                                                        #conversation history is maintained as a thread


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)


