import asyncio
from typing import TypedDict, List, Union

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

class GraphState(TypedDict):
    history: List[BaseMessage]

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="hermes-3-llama-3.2-3b",
    streaming=True,
)

def init_state() -> GraphState:
    return {"history": []}

def get_input(state: GraphState) -> Union[str, GraphState]:
    text = input("You: ").strip()
    if text.lower() == "quit":
        return END
    state["history"].append(HumanMessage(content=text))
    return state

async def respond(state: GraphState) -> GraphState:
    response = ""
    async for chunk in llm.astream(state["history"]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            response += chunk.content
    print()
    state["history"].append(AIMessage(content=response))
    return state

builder = StateGraph(GraphState)
builder.add_node("get_input", get_input)
builder.add_node("respond", RunnableLambda(respond))
builder.set_entry_point("get_input")
builder.add_edge("get_input", "respond")
builder.add_edge("respond", "get_input")
graph = builder.compile()

if __name__ == "__main__":
    asyncio.run(graph.ainvoke(init_state()))
