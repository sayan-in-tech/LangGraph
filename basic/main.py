import asyncio
from typing import TypedDict, List, Union

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

class GraphState(TypedDict):
    history: List[BaseMessage]
    continue_: bool

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="hermes-3-llama-3.2-3b",
    streaming=True,
)

async def chat_node(state: GraphState) -> GraphState:
    text = input("You: ").strip()
    if text.lower() == "quit":
        return {"history": state["history"], "continue_": False}

    state["history"].append(HumanMessage(content=text))

    reply = ""
    async for chunk in llm.astream(state["history"]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            reply += chunk.content
    print()

    state["history"].append(AIMessage(content=reply))
    return {"history": state["history"], "continue_": True}

# Define the graph
builder = StateGraph(GraphState)
builder.add_node("chat", RunnableLambda(chat_node))
builder.set_entry_point("chat")

# Use a conditional edge to end the graph
builder.add_conditional_edges(
    "chat",
    lambda state: "chat" if state["continue_"] else None
)

graph = builder.compile()

if __name__ == "__main__":
    asyncio.run(graph.ainvoke({"history": [], "continue_": True}))
