import os
import operator
from typing import TypedDict, Annotated, List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# Import our custom modules
from state import AgentState
from tools import (
    claim_parsing_agent,
    deep_evidence_retrieval_agent,
    reasoning_explanation_agent,
    critic_calibration_agent,
    final_reporting_agent
)
from orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT

# Load environment variables (ensure .env exists with OPENAI_API_KEY)
load_dotenv()

# 1. Setup Tools
tools = [
    claim_parsing_agent,
    deep_evidence_retrieval_agent,
    reasoning_explanation_agent,
    critic_calibration_agent,
    final_reporting_agent
]

# 2. Setup LLM (Orchestrator)
# Using GPT-4o or similar high-reasoning model is recommended for orchestration
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 3. Define Orchestrator Node
def orchestrator_node(state: AgentState):
    """
    The Orchestrator node uses the history of messages to decide the next step.
    It uses the specific system prompt to govern the workflow.
    """
    messages = state["messages"]
    
    # Ensure system prompt is the first message
    # In a real app, you might handle this initialization differently
    prompt_message = SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)
    
    # We construct the call. If messages is empty (start), add system prompt.
    # If not empty, we need to ensure the model sees the system prompt.
    # LangChain ChatOpenAI usually handles 'system' roles well.
    
    if not messages or not isinstance(messages[0], SystemMessage):
         # If the state doesn't have the system prompt, we logically prepend it for the API call
         # but maybe not persist it to state if we want to extract it later.
         # For simplicity, let's just invoke with [System, ...messages]
         invocation_messages = [prompt_message] + messages
    else:
         invocation_messages = messages
         
    # Invoke the LLM
    response = llm_with_tools.invoke(invocation_messages)
    
    # Return the new AI Message (update state)
    return {"messages": [response]}

# 4. Define Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("tools", ToolNode(tools))

# Add edges
workflow.add_edge(START, "orchestrator")

def should_continue(state: AgentState):
    """
    Determine if we should continue to tools or end.
    """
    last_message = state["messages"][-1]
    
    # If the LLM called a tool, route there
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, if it returned text (and hopefully the final JSON), end
    # Ideally, we'd validate the JSON structure here.
    return END

workflow.add_conditional_edges(
    "orchestrator",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# After tools execute, go back to orchestrator to reason on the output
workflow.add_edge("tools", "orchestrator")

# Compile
app = workflow.compile()

# 5. Example Execution
if __name__ == "__main__":
    print("--- Antigravity Orchestrator Initialized ---")
    
    user_input = "Vaccines can cause autism."
    print(f"User Input: {user_input}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "input_text": user_input
    }
    
    # Run the graph
    # Using 'stream' to see steps
    try:
        dict_inputs = initial_state
        for event in app.stream(dict_inputs):
            for key, value in event.items():
                print(f"\n[Node: {key}]")
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        if last_msg.tool_calls:
                            print(f"Orchestrator decided to call: {len(last_msg.tool_calls)} tools")
                            for tc in last_msg.tool_calls:
                                print(f" - {tc['name']}")
                        else:
                            print(f"Orchestrator Output: {last_msg.content}")
                    elif isinstance(last_msg, ToolMessage):
                        print(f"Tool Result: {last_msg.content[:100]}...")
    except Exception as e:
        print(f"Error running graph: {e}")
        print("Note: Ensure OPENAI_API_KEY is set in .env")

