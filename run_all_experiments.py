import subprocess
import os
import time
from typing import List, Dict

# List of models to evaluate
MODELS_TO_TEST = [
    # OpenAI Models
    {"model": "gpt-4o-mini", "provider": "openai"},
    {"model": "gpt-4o", "provider": "openai"},
    
    # OpenRouter Free Models (Representative set from screenshots)
    {"model": "meta-llama/llama-3.3-70b-instruct:free", "provider": "openrouter"},
    {"model": "deepseek/deepseek-r1:free", "provider": "openrouter"},
    {"model": "mistralai/mistral-small-24b-instruct-2501:free", "provider": "openrouter"},
    {"model": "google/gemma-3-27b:free", "provider": "openrouter"},
    {"model": "google/gemma-3-12b:free", "provider": "openrouter"},
    {"model": "mistralai/mistral-small-24b-instruct-2501:free", "provider": "openrouter"},
    {"model": "deepseek/deepseek-r1:free", "provider": "openrouter"},
    {"model": "nousresearch/hermes-3-llama-3.1-405b:free", "provider": "openrouter"},
    {"model": "qwen/qwen-3-4b:free", "provider": "openrouter"},
    {"model": "nvidia/nemotron-4-340b-instruct:free", "provider": "openrouter"},
    {"model": "liquid/lfm-2.5-1.2b-instruct:free", "provider": "openrouter"},
    {"model": "meta-llama/llama-3.2-3b-instruct:free", "provider": "openrouter"},
    {"model": "anthropic/claude-3.5-sonnet", "provider": "openrouter"},
    # Example for Ollama (ensure Ollama is running)
    {"model": "llama3.1", "provider": "ollama", "base_url": "http://localhost:11434/v1"}
]

# Experiments to run for each model
# Use "all" to run all configurations defined in ablation_study.py
# Or list specific ones like ["Baseline", "PubMed_Only", "CoT_Orchestrator"]
EXPERIMENTS = "all" 

def run_experiment(model_name: str, provider: str, base_url: str = None, test_mode: bool = False, max_examples: int = 50):
    """Executes the ablation study script for a specific model."""
    cmd = [
        "python", "ablation_study.py",
        "--model", model_name,
        "--model-provider", provider,
        "--experiment", EXPERIMENTS,
        "--max-examples", str(max_examples)
    ]
    
    if base_url:
        cmd.extend(["--model-base-url", base_url])
    
    if test_mode:
        cmd.append("--test-mode")
        # Optional: cmd.extend(["--max-examples", "5"]) # Uncomment to run more than 2 but not all
    
    print(f"\n" + "="*80)
    print(f"🚀 STARTING EXPERIMENTS FOR MODEL: {model_name} ({provider})")
    print(f"Command: {' '.join(cmd)}")
    print("="*80 + "\n")
    
    try:
        # Run and stream output to terminal
        process = subprocess.Popen(cmd, stdout=None, stderr=None)
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ Successfully completed experiments for {model_name}\n")
        else:
            print(f"\n❌ Experiments for {model_name} failed with return code {process.returncode}\n")
    except Exception as e:
        print(f"\n❌ Error running experiment for {model_name}: {e}\n")

def main():
    print(f"Starting Multi-Model Evaluation Runner")
    print(f"Time: {time.ctime()}")
    print(f"Models to evaluate: {[m['model'] for m in MODELS_TO_TEST]}")
    
    # Set test_mode=False for a full production run
    TEST_MODE = False 
    MAX_EXAMPLES = 50
    
    for entry in MODELS_TO_TEST:
        run_experiment(
            model_name=entry["model"],
            provider=entry["provider"],
            base_url=entry.get("base_url"),
            test_mode=TEST_MODE,
            max_examples=MAX_EXAMPLES
        )
    
    print("\n" + "#"*80)
    print("ALL MODEL EVALUATIONS COMPLETE")
    print("#"*80)

if __name__ == "__main__":
    main()
