import argparse
import sys
from main import create_workflow
from evaluate_workflow import run_evaluation
from tools import (
    claim_parsing_tool, evidence_retrieval_tool, reasoning_explanation_tool, 
    critic_calibration_tool, final_reporting_tool, pubmed_search_tool, 
    cross_lingual_bundle_tool
)

def get_configs():
    """
    Define the ablation configurations.
    """
    base_tools = [
        claim_parsing_tool, evidence_retrieval_tool, reasoning_explanation_tool, 
        critic_calibration_tool, final_reporting_tool, pubmed_search_tool, 
        cross_lingual_bundle_tool
    ]

    # Configuration Definitions
    configs = {
        "Baseline": {
            "config": {
                "use_umls": True, "use_filtering": True, "use_prioritization": True, 
                "use_span_extraction": True, "use_calibration": True, 
                "use_recency": True, "use_faithfulness": True, "use_debate": True
            },
            "tools": base_tools
        },
        "A1_No_UMLS": {
            "config": {
                "use_umls": False, "use_filtering": True, "use_prioritization": True, 
                "use_span_extraction": True, "use_calibration": True, 
                "use_recency": True, "use_faithfulness": True, "use_debate": True
            },
            "tools": base_tools
        },
         "A2_No_Filtering": {
            "config": {
                "use_umls": True, "use_filtering": False, "use_prioritization": True, 
                "use_span_extraction": True, "use_calibration": True, 
                "use_recency": True, "use_faithfulness": True, "use_debate": True
            },
            "tools": base_tools
        },
        "A3_No_Prioritization": {
            "config": {
                "use_umls": True, "use_filtering": True, "use_prioritization": False, 
                "use_span_extraction": True, "use_calibration": True, 
                "use_recency": True, "use_faithfulness": True, "use_debate": True
            },
            "tools": base_tools
        },
        "A3_No_Calibration": {
            "config": {
                "use_umls": True, "use_filtering": True, "use_prioritization": True, 
                "use_span_extraction": True, "use_calibration": False, 
                "use_recency": True, "use_faithfulness": True, "use_debate": True
            },
            "tools": base_tools
        },
        "A4_No_Critic_Agent": {
            "config": {
                # Config flags don't matter much if the tool is removed, but for consistency:
                "use_umls": True, "use_filtering": True, "use_prioritization": True, 
                "use_span_extraction": True, "use_calibration": True, 
                "use_recency": False, "use_faithfulness": False, "use_debate": False,
                "use_critic": False
            },
            # Remove Critic Tool? NO. Keep it as dummy to prevent Prompt Confusion.
            "tools": base_tools
        },
        "A4_No_Debate": {
            "config": {
                "use_umls": True, "use_filtering": True, "use_prioritization": True, 
                "use_span_extraction": True, "use_calibration": True, 
                "use_recency": True, "use_faithfulness": True, "use_debate": False
            },
            "tools": base_tools
        }
    }
    return configs

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies on the Misinformation Workflow.")
    parser.add_argument("--test-mode", action="store_true", help="Run on a tiny subset (2 examples) for verification.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit the number of examples to evaluate.")
    parser.add_argument("--experiment", type=str, default="all", help="Specific experiment to run (key name) or 'all'.")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model name (e.g., 'gpt-3.5-turbo', 'llama3').")
    parser.add_argument("--model-base-url", type=str, default=None, help="Override LLM base URL (e.g., 'http://localhost:11434/v1').")
    parser.add_argument("--model-provider", type=str, default="openai", choices=["openai", "ollama"], help="Specify LLM provider (openai or ollama). Default: openai")
    args = parser.parse_args()

    configs = get_configs()
    dataset_name = "Health_Misinformation_Eval_Subset"
    
    experiments_to_run = list(configs.keys()) if args.experiment == "all" else [args.experiment]

    print(f"Starting Ablation Study. Experiments: {experiments_to_run}")

    for exp_name in experiments_to_run:
        if exp_name not in configs:
            print(f"Error: Experiment '{exp_name}' not found. Available: {list(configs.keys())}")
            continue

        print(f"\n{'='*50}")
        print(f"Running Experiment: {exp_name}")
        print(f"{'='*50}")

        settings = configs[exp_name]
        
        # Construct shared LLM config if args provided
        cli_llm_config = {}
        if args.model:
            cli_llm_config["model_name"] = args.model
        if args.model_base_url:
            cli_llm_config["base_url"] = args.model_base_url
        if args.model_provider:
            cli_llm_config["model_provider"] = args.model_provider
            
        # Merge with experiment specific (if any)
        exp_llm_config = settings.get("llm_config", {})
        final_llm_config = {**exp_llm_config, **cli_llm_config}
        
        # 1. Create Workflow
        app_instance = create_workflow(config=settings["config"], tools_list=settings["tools"], llm_config=final_llm_config)
        
        # 2. Run Evaluation
        # We append a timestamp or unique ID in LangSmith via the key
        experiment_prefix = f"ablation-{exp_name}"
        if args.model: 
            experiment_prefix += f"-{args.model}"
        
        try:
            # Priority: Command line arg > Test mode default (2) > None (All)
            if args.max_examples is not None:
                max_ex = args.max_examples
            elif args.test_mode:
                max_ex = 2
            else:
                max_ex = None
                
            results = run_evaluation(app_instance, dataset_name, experiment_prefix, max_examples=max_ex)
            print(f"Experiment {exp_name} complete. Results: {results}")
        except Exception as e:
            print(f"Experiment {exp_name} FAILED: {e}")

if __name__ == "__main__":
    main()
