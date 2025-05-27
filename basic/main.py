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

async def single_node(state: GraphState) -> Union[str, GraphState]:
    text = input("You: ").strip()
    if text.lower() == "quit":
        return 0
    state["history"].append(HumanMessage(content=text))

    reply = ""
    async for chunk in llm.astream(state["history"]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            reply += chunk.content
    print()
    state["history"].append(AIMessage(content=reply))
    return state

builder = StateGraph(GraphState)
builder.add_node("chat", RunnableLambda(single_node))
builder.set_entry_point("chat")
builder.add_edge("chat", "chat")
graph = builder.compile()

if __name__ == "__main__":
    asyncio.run(graph.ainvoke({"history": []}))
