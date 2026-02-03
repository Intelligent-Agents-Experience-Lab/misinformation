import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from main import create_workflow

# Load env vars
load_dotenv()

def test_ollama_execution():
    print("--- Starting Debug Test for Ollama (llama3.2:1b) ---")
    
    # Configuration for Ollama
    llm_config = {
        "model_name": "llama3.2:1b",
        "model_provider": "ollama",
        "base_url": "http://localhost:11434" # Default Ollama URL
    }
    
    # Create workflow
    app = create_workflow(llm_config=llm_config)
    
    # Test Input
    user_input = "Vaccines can cause the diseases they prevent."
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "input_text": user_input
    }
    
    print(f"Input: {user_input}")
    
    # Run loop
    step_count = 0
    start_time = time.time()
    
    try:
        for event in app.stream(initial_state):
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            for key, value in event.items():
                print(f"Node: {key}")
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    print(f"Message Type: {type(last_msg).__name__}")
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"Tool Calls: {last_msg.tool_calls}")
                    if hasattr(last_msg, "content"):
                        print(f"Content: {last_msg.content[:200]}...") # Truncate
                        
            if time.time() - start_time > 60:
                print("!!! Timeout reached (60s) !!!")
                break
                
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_ollama_execution()
