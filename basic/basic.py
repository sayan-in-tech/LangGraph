from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
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
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Define the graph structure

graph_builder.add_node(
    "chatbot",
    chatbot
)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# graph_builder.add_node(
#     "chatbot",
#     chatbot
# )

# graph_builder.set_entry_point(
#     "chatbot"
# )
# graph_builder.set_finish_point(
#     "chatbot"
# )

# graph_builder.add_node(
#     START,
#     chatbot
# )
# graph_builder.add_node(
#     "chatbot",
#     chatbot
# )
# graph_builder.add_node(
#     "chatbot",
#     END
# )

# Build the graph
graph = graph_builder.compile()

while True:
    user_input = input("Enter your message: ")

    if user_input.lower() in ["exit", "quit", "bye", "/bye"]:
        print("Exiting the chatbot.")
        break
    state = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }
    )

    print("Chatbot response:", state["messages"][-1].content)
