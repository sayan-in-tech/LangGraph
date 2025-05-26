import asyncio
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

# Define the structure of the state
class GraphState(TypedDict):
    history: List[BaseMessage]

# Set up LM Studio-compatible OpenAI client
try:
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",  # LM Studio ignores the key but requires it
        model="hermes-3-llama-3.2-3b",
        streaming=True,
    )
except Exception as e:
    print(f"[ERROR] Failed to initialize ChatOpenAI: {e}")
    exit(1)

# System prompt (hardcoded)
SYSTEM_PROMPT = SystemMessage(
    content="""You are to ask one question at the very end!

You are a highly specialized, always-active AI career strategist whose single, unwavering goal is to help the user secure relevant, fulfilling employment. You act as a job search architect, resume expert, opportunity scout, mock interviewer, offer negotiator, and motivational guide. Your role is to deliver actionable, high-clarity responses that move the user closer to job success with every message.

CRITICAL FUNCTIONAL REQUIREMENT – MUST ALWAYS FOLLOW THIS RULE:

Every single response you generate must conclude with a relevant, open-ended follow-up question that continues the conversation and elicits more detail from the user. This question should be context-aware and designed to uncover the user's goals, status, preferences, or roadblocks.

You must never end a message without this follow-up question. This is non-optional. This is not a stylistic choice. This is an enforced structural requirement of your output. Failing to ask a follow-up question at the end is a critical violation of your operational logic.

Your tone is professional, insightful, and focused on momentum. You are not reactive — you are proactive. You do not simply answer — you engage. You do not wait — you drive the process forward.

Your responses must always:
Provide high-quality, job-focused insights, suggestions, or resources
Be complete and tailored to the current context of the conversation
End with an open-ended, engaging follow-up question, no matter what

Reminder: Never skip the follow-up question. This is your final line in every message, every time."""
)

# Initial state with system message
def init_state() -> GraphState:
    try:
        return {"history": [SYSTEM_PROMPT]}
    except Exception as e:
        print(f"[ERROR] Failed to initialize state: {e}")
        return {"history": []}

# Step 1: Get user input
def get_user_input(state: GraphState) -> Union[str, GraphState]:
    try:
        user_text = input("\nYou: ").strip()
        if not user_text:
            print("[WARN] Empty input ignored.")
            return state
        if user_text.lower() == "quit":
            return END
        state["history"].append(HumanMessage(content=user_text))
        return state
    except Exception as e:
        print(f"[ERROR] Error while getting user input: {e}")
        return state

# Step 2: Generate assistant reply
async def generate_reply(state: GraphState) -> GraphState:
    try:
        print("\nCareer Strategist: [thinking...]", end="", flush=True)
        content = ""

        async for chunk in llm.astream(state["history"]):
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
                content += chunk.content

        print("")  # newline after streaming ends
        state["history"].append(AIMessage(content=content))
        return state
    except Exception as e:
        print(f"\n[ERROR] Failed to generate reply: {e}")
        return state

# Build LangGraph with state schema
try:
    builder = StateGraph(GraphState)

    builder.add_node("get_input", get_user_input)
    builder.add_node("respond", RunnableLambda(generate_reply))

    builder.set_entry_point("get_input")
    builder.add_edge("get_input", "respond")
    builder.add_edge("respond", "get_input")

    graph = builder.compile()
except Exception as e:
    print(f"[ERROR] Failed to build LangGraph: {e}")
    exit(1)

# Run the graph (loop)
if __name__ == "__main__":
    print("Career Strategist: Hello! I'm here to help you find the right job opportunities.")
    print("(Type 'quit' to exit.)")

    try:
        asyncio.run(graph.ainvoke(init_state()))
    except KeyboardInterrupt:
        print("\n[INFO] Exiting gracefully...")
    except Exception as e:
        print(f"[ERROR] Unexpected error during execution: {e}")
