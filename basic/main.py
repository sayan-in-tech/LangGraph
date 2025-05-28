import asyncio
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="hermes-3-llama-3.2-3b",
    streaming=True  # Set to False unless you're handling streaming responses
)

# Define the conversation state
class State(TypedDict):
    messages: List[BaseMessage]
    message_type: str | None

# Chat function that takes in a state and returns updated state
def chat(state: State) -> State:
    last_message = state["messages"][-1]
    messages = state["messages"]
    
    response = llm.invoke(messages)
    
    # Append assistant response
    updated_messages = messages + [response]
    
    return {
        "messages": updated_messages,
        "message_type": "ai"
    }

# Define the graph
graph_builder = StateGraph(State)

graph_builder.add_node("chat", RunnableLambda(chat))
graph_builder.set_entry_point("chat")
graph_builder.set_finish_point("chat")

graph = graph_builder.compile()

# Run the chatbot
def run_chatbot():
    state: State = {
        "messages": [],
        "message_type": None
    }

    while True:
        user_input = input("Message: ")
        if user_input.lower() in {"exit", "quit", "/bye"}:
            print("Bye")
            break

        # Add user message
        state["messages"].append(HumanMessage(content=user_input))

        # Invoke graph
        state = graph.invoke(state)

        # Display assistant message
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()
