# Health Misinformation Verification Workflow

A multi-agent, automated pipeline for verifying health-related claims, integrating advanced retrieval, reasoning, and evaluation metrics via LangSmith and LangGraph.

## 🚀 Overview

This project implements an agentic workflow to detect and debunk health misinformation. It uses an orchestrator-led architecture to parse claims, retrieve medical evidence (including PubMed), reason over the findings, and generate structured reports with confidence scores.

## 🏗️ Architecture

The system utilizes a multi-step graph execution:
- **Claim Parsing**: Decomposes user input into verifiable claims.
- **Evidence Retrieval**: Fetches grounded evidence from web search and PubMed.
- **Reasoning**: Analyzes claims against evidence using LLM-as-a-judge reasoning.
- **Critic & Calibration**: Evaluates reasoning for bias, recency, and faithfulness.
- **Final Reporting**: Produces a clean, production-ready JSON verification report.

## 📊 Evaluation Suite

The project includes a robust evaluation pipeline (`evaluate_workflow.py`) with **six core agentic metrics** ported from DeepEval to LangSmith:

1.  **Task Completion**: Assesses if the final output semantically fulfills the user's request.
2.  **Tool Correctness**: Verifies if the agent used the right tools in a logical order.
3.  **Argument Correctness**: Evaluates the precision and relevance of tool-calling arguments.
4.  **Step Efficiency**: Penalizes redundant tool calls and rewards direct paths to the answer.
5.  **Plan Adherence**: Checks if the agent followed its initially stated reasoning/plan.
6.  **Plan Quality**: Judges the logic, completeness, and actionability of the agent's plan.

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API Key
- LangSmith API Key (for evaluation tracking)

### Installation
```bash
pip install -r requirements.txt
```

### Running the Workflow of signle Agent
```python
python main.py
```


### Running the Workflow of multi Agent
```python
python multi_agent_main.py
```

### Running Evaluations
To run a full holistic evaluation:
```bash
python evaluate_workflow.py
```

To run an **Ablation Study** (testing different configurations):
```bash
python ablation_study.py --experiment Baseline --test-mode
```

## 🧪 Ablation Study Configuration

You can toggle various features in `ablation_study.py`:
- `use_umls`: Biological entity normalization.
- `use_filtering`: Evidence relevance filtering.
- `use_prioritization`: Source ranking.
- `use_calibration`: Confidence scoring.
- `use_debate`: Multiple reasoning paths.

## 📈 Monitoring
All evaluation results are automatically logged to **LangSmith**, providing granular trace analysis and per-metric scoring for every execution.
