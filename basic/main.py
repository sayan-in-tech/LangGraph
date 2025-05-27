import asyncio
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
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

async def chat_step(state: GraphState) -> GraphState:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        return {"history": state["history"], "continue_": False}

    state["history"].append(HumanMessage(content=user_input))

    reply = ""
    async for chunk in llm.astream(state["history"]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            reply += chunk.content
    print()

    state["history"].append(AIMessage(content=reply))
    return {"history": state["history"], "continue_": True}

builder = StateGraph(GraphState)
builder.add_node("chat", RunnableLambda(chat_step))
builder.set_entry_point("chat")
builder.add_conditional_edges("chat", lambda state: "chat" if state["continue_"] else END)
graph = builder.compile()

if __name__ == "__main__":
    asyncio.run(graph.ainvoke({"history": [], "continue_": True}))
