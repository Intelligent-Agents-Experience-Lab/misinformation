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
    claim_parsing_tool,
    evidence_retrieval_tool,
    reasoning_explanation_tool,
    critic_calibration_tool,
    final_reporting_tool,
    pubmed_search_tool,
    filtering_and_dedup_tool,
    cross_lingual_bundle_tool,
    evidence_prioritization_tool,
    verbatim_span_extraction_tool,
    logistic_calibration_tool,
    span_faithfulness_tool,
    recency_decay_tool,
    confidence_adjustment_tool,
    debate_adjudicator_tool,
    set_ablation_config  # Import config setter
)
from orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT


# Load environment variables (ensure .env exists with OPENAI_API_KEY)
load_dotenv()

# Factory function for creating the workflow dynamically
def create_workflow(config: dict = None, tools_list: list = None, llm_config: dict = None):
    """
    Creates and compiles the StateGraph workflow.
    Args:
        config (dict): Optional ablation configuration to override defaults.
        tools_list (list): Optional list of tools to use (defaults to full set).
        llm_config (dict): Optional LLM settings (model_name, base_url, api_key).
    """
    
    # 1. Update Global Ablation Config if provided
    if config:
        set_ablation_config(config)
        print(f"--- Workflow Created with Config: {config} ---")

    # 2. Setup Tools
    # Default set of tools if not provided
    if tools_list is None:
        tools_list = [
            claim_parsing_tool,
            evidence_retrieval_tool,
            reasoning_explanation_tool,
            critic_calibration_tool,
            final_reporting_tool,
            pubmed_search_tool,
            cross_lingual_bundle_tool
        ]
    
    # 3. Setup LLM (Orchestrator)
    # Defaults
    model_name = "gpt-4o-mini"
    kwargs = {"temperature": 0}
    
    if llm_config:
        print(f"--- Initializing Orchestrator with LLM: {llm_config} ---")
        if "model_name" in llm_config:
            model_name = llm_config["model_name"]
        if "base_url" in llm_config:
            kwargs["base_url"] = llm_config["base_url"]
        if "api_key" in llm_config:
            kwargs["api_key"] = llm_config["api_key"]
    
    llm = ChatOpenAI(model=model_name, **kwargs)
    llm_with_tools = llm.bind_tools(tools_list)

    # 4. Define Orchestrator Node
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

    # 5. Define Graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("tools", ToolNode(tools_list))
    
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
    return workflow.compile()

# Default app instance for backward compatibility
app = create_workflow()

# 6. Example Execution
if __name__ == "__main__":
    print("--- Orchestrator Initialized ---")
    
    user_input = "Social media can be a tool for health promotion and education."
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

