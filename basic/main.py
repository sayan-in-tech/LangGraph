import asyncio
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, add_messages
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="hermes-3-llama-3.2-3b",
    streaming=True,
)

class State(TypedDict):
    messages: Annotated[List, add_messages]
    message_type: str | None

def chat(state: State) -> State:
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {
        "messages": [{"role": "user", "content": last_message.content}]}
                     
graph_builder = StateGraph(State)

graph_builder.add_node(START, chat)
graph_builder.add_node("chat", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input.lower() == "exit" or user_input.lower() == "quit" or user_input.lower() == "/bye":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()