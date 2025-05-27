import asyncio
from typing import TypedDict, List, Union

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

# Define the conversation state type
class GraphState(TypedDict):
    history: List[BaseMessage]

# Initialize LM Studio LLM client
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",  # LM Studio expects a key but does not validate
    model="hermes-3-llama-3.2-3b",
    streaming=True,
)

# Base system prompt (replace as needed)
SYSTEM_PROMPT = SystemMessage(content="You are a helpful assistant.")

# Initialize conversation state
def init_state() -> GraphState:
    return {"history": [SYSTEM_PROMPT]}

# Step 1: get user input
def get_user_input(state: GraphState) -> Union[str, GraphState]:
    user_text = input("\nYou: ").strip()
    if user_text.lower() == "quit":
        return END
    state["history"].append(HumanMessage(content=user_text))
    return state

# Step 2: generate AI response (streaming)
async def generate_reply(state: GraphState) -> GraphState:
    collected = ""
    stream = llm.astream(state["history"])
    async for chunk in stream:
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
            collected += chunk.content
    print()
    state["history"].append(AIMessage(content=collected))
    return state

# Build LangGraph state machine
builder = StateGraph(GraphState)
builder.add_node("get_input", get_user_input)
builder.add_node("respond", RunnableLambda(generate_reply))
builder.set_entry_point("get_input")
builder.add_edge("get_input", "respond")
builder.add_edge("respond", "get_input")
graph = builder.compile()

# Run main loop
if __name__ == "__main__":
    print("Assistant started. Type 'quit' to exit.")
    asyncio.run(graph.ainvoke(init_state()))
