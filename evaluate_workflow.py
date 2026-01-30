import os
import evaluate
import json
import time
import re
from datasets import load_dataset
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from main import app
from dotenv import load_dotenv

load_dotenv()

client = Client()

# Initialize HF Evaluate metrics
print("Loading HF Evaluate metrics...")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

# 1. Load Dataset
print("Loading Health_Misinformation dataset...")
ds = load_dataset("ClassyB/Health_Misinformation", split="train")
subset = ds.select(range(10))

# 2. Prepare LangSmith Dataset
dataset_name = "Health_Misinformation_Eval_Subset"
try:
    langsmith_dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="Subset of 10 examples from Health_Misinformation for verification workflow testing."
    )
except Exception:
    langsmith_dataset = client.read_dataset(dataset_name=dataset_name)

examples = []
for item in subset:
    examples.append({
        "inputs": {"question": item["Statement"]},
        "outputs": {"answer": item["Reason"], "label": item["True or Misinformation"]},
    })

if langsmith_dataset.example_count == 0:
    client.create_examples(dataset_id=langsmith_dataset.id, examples=examples)

# 3. Define Target Function
def target(inputs: dict) -> dict:
    user_input = inputs["question"]
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "input_text": user_input
    }
    
    start_time = time.perf_counter()
    tools_called = []
    evidence_context = ""
    entities_found = []
    final_output_str = ""
    
    try:
        for event in app.stream(initial_state):
             for key, value in event.items():
                if key == "tools":
                    # Capture tool data and entities
                    for msg in value.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            # Try to parse entities if it's the parsing tool
                            try:
                                tool_result = json.loads(msg.content)
                                if "claims" in tool_result:
                                    for claim in tool_result["claims"]:
                                        if "entities" in claim:
                                            entities_found.extend([e["preferred"] for e in claim["entities"]])
                                if "Source:" in msg.content or "snippet" in msg.content:
                                    # Fallback for search results if content is flat string
                                    evidence_context += msg.content + "\n"
                            except:
                                if "Source:" in msg.content:
                                    evidence_context += msg.content + "\n"
                
                if key == "orchestrator":
                    last_msg = value["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        if last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                tools_called.append(tc['name'])
                        else:
                            final_output_str = last_msg.content
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "tools_called": tools_called, "latency_sec": time.perf_counter() - start_time}

    end_time = time.perf_counter()
    return {
        "answer": final_output_str, 
        "tools_called": tools_called, 
        "evidence": evidence_context,
        "entities": list(set(entities_found)),
        "latency_sec": end_time - start_time
    }

# 4. Define Evaluators

def health_correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    return evaluator(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

def hf_statistical_evaluators(inputs: dict, outputs: dict, reference_outputs: dict):
    prediction = outputs.get("answer", "")
    reference = reference_outputs.get("answer", "")
    if not prediction or not reference: return {"results": []}
    
    bleu = bleu_metric.compute(predictions=[prediction], references=[reference])["bleu"]
    rouge = rouge_metric.compute(predictions=[prediction], references=[reference])["rougeL"]
    meteor = meteor_metric.compute(predictions=[prediction], references=[reference])["meteor"]
    
    return {
        "results": [
            {"key": "hf_bleu", "score": bleu},
            {"key": "hf_rougeL", "score": rouge},
            {"key": "hf_meteor", "score": meteor}
        ]
    }

def performance_evaluators(inputs: dict, outputs: dict, reference_outputs: dict):
    latency = outputs.get("latency_sec", 0)
    entities = outputs.get("entities", [])
    answer = outputs.get("answer", "")
    
    # Entity Consistency Score
    addressed_count = 0
    if entities:
        for entity in entities:
            if entity.lower() in answer.lower():
                addressed_count += 1
        consistency_score = addressed_count / len(entities)
    else:
        consistency_score = 1.0 # No entities to cover
        
    # Evidence Density
    urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', answer)
    unique_sources = len(set(urls))
    
    return {
        "results": [
            {"key": "latency_seconds", "score": latency},
            {"key": "entity_consistency", "score": consistency_score},
            {"key": "unique_source_count", "score": float(unique_sources)}
        ]
    }

def medical_grounding_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    reasoning = outputs.get("answer", "")
    evidence = outputs.get("evidence", "")
    
    from langchain_openai import ChatOpenAI
    eval_llm = ChatOpenAI(model="o3-mini")
    
    prompt = f"""Assess if the following medical reasoning is grounded in the provided search evidence.
Reasoning: {reasoning}
Evidence: {evidence}

Respond with a score between 0 and 1, where 1 means perfectly grounded and 0 means no grounding or contradicts evidence.
Provide only the numeric score."""
    
    try:
        response = eval_llm.invoke(prompt)
        score = float(response.content.strip())
        return {"key": "medical_grounding", "score": score}
    except:
        return {"key": "medical_grounding", "score": 0.0}

# 5. Run Evaluation
print(f"Starting evaluation on dataset: {dataset_name} with Full Holistic metrics...")
experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[
        health_correctness_evaluator,
        hf_statistical_evaluators,
        performance_evaluators,
        medical_grounding_evaluator
    ],
    experiment_prefix="misinfo-workflow-holistic-eval",
    max_concurrency=1,
)

print("\nEvaluation complete!")
print(f"View results in LangSmith: {experiment_results}")
