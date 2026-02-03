import os
import evaluate
import json
import time
import re
from datasets import load_dataset
from langsmith import Client
from langchain_openai import ChatOpenAI
# from openevals.llm import create_llm_as_judge
# from openevals.prompts import CORRECTNESS_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from langchain.evaluation import LangChainStringEvaluator # causing import issues
from main import app as default_app
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
subset = ds.select(range(5))

# 2. Prepare LangSmith Dataset
dataset_name = "Health_Misinformation_Eval_Subset"
try:
    langsmith_dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Subset of 300 examples from Health_Misinformation for verification workflow testing."
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


# 4. Define Evaluators

def health_correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    # Custom LLM-as-a-judge implementation since openevals is missing
    judge_llm = ChatOpenAI(model="o3-mini")

    question = inputs.get("question", "")
    prediction = outputs.get("answer", "")
    reference_reason = reference_outputs.get("answer", "")
    reference_label = reference_outputs.get("label", "")

    prompt = f"""You are an expert medical misinformation evaluator.
    Compare the Model's Response to the Reference Answer.

    Question: {question}

    Reference Label: {reference_label}
    Reference Reason: {reference_reason}

    Model Response: {prediction}

    Task:
    1. Verify if the Model's label (True/Misinformation) matches the Reference Label.
    2. Verify if the Model's explanation captures the key points of the Reference Reason.
    3. Ignore minor phrasing differences. Focus on semantic accuracy.

    Return a JSON object with:
    - score: (0 to 1, where 1 is correct)
    - contrast: (Explanation of differences)
    """

    try:
        response = judge_llm.invoke(prompt)
        # Simple parsing if model returns raw JSON string
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return {"key": "correctness", "score": result.get("score", 0), "comment": result.get("contrast", "")}
    except Exception as e:
        return {"key": "correctness", "score": 0, "comment": f"Eval Error: {str(e)}"}

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

def run_evaluation(app_instance, dataset_name, experiment_prefix="misinfo-eval", max_examples=None):
    """
    Reusable evaluation function.
    Args:
        max_examples (int): Optional limit on number of examples to evaluate.
    """
    print(f"Starting evaluation: {experiment_prefix} on dataset: {dataset_name} (Max Examples: {max_examples})")

    print(f"Starting evaluation: {experiment_prefix} on dataset: {dataset_name} (Max Examples: {max_examples})")

    # Define Custom Evaluators due to import issues
    def conciseness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""Evaluate the conciseness of the following response.
        Response: {outputs.get("answer", "")}
        
        Is the response concise and to the point?
        Return a score between 0 and 1 (1 = very concise)."""
        try:
            res = eval_llm.invoke(prompt).content
            score = float(re.search(r"[\d\.]+", res).group()) if re.search(r"[\d\.]+", res) else 0.5
            return {"key": "conciseness", "score": score}
        except: return {"key": "conciseness", "score": 0.5}

    def coherence_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""Evaluate the coherence of the following response.
        Response: {outputs.get("answer", "")}
        
        Is the response coherent and well-structured?
        Return a score between 0 and 1 (1 = very coherent)."""
        try:
            res = eval_llm.invoke(prompt).content
            score = float(re.search(r"[\d\.]+", res).group()) if re.search(r"[\d\.]+", res) else 0.5
            return {"key": "coherence", "score": score}
        except: return {"key": "coherence", "score": 0.5}

    def relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""Evaluate the relevance of the response to the input question.
        Question: {inputs.get("question", "")}
        Response: {outputs.get("answer", "")}
        
        Return a score between 0 and 1 (1 = highly relevant)."""
        try:
            res = eval_llm.invoke(prompt).content
            score = float(re.search(r"[\d\.]+", res).group()) if re.search(r"[\d\.]+", res) else 0.5
            return {"key": "relevance", "score": score}
        except: return {"key": "relevance", "score": 0.5}
        
    # Placeholder for embedding distance (skipping complex dependency for now)
    def embedding_distance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        # Mock or simplified string distance if embedding model unavailable
        # We'll use a simple Levenshtein ratio proxy or just skip if no embeddings
        return {"key": "embedding_distance", "score": 0.0} # Placeholder

    # 3. Define Target Function (Closure to capture app_instance)
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
            for event in app_instance.stream(initial_state):
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

        try:
            if "```json" in final_output_str:
                clean_json = final_output_str.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
            else:
                data = json.loads(final_output_str)

            # Extract fields from the first claim
            if "claims" in data and len(data["claims"]) > 0:
                claim_data = data["claims"][0]
                answer_text = f"Label: {claim_data.get('label')}\nExplanation: {claim_data.get('explanation')}"
                predicted_label = claim_data.get("label")
            else:
                answer_text = final_output_str
                predicted_label = "Unknown"
        except:
            answer_text = final_output_str
            predicted_label = "Parse Error"

        end_time = time.perf_counter()
        return {
            "answer": answer_text,
            "predicted_label": predicted_label,
            "tools_called": tools_called,
            "evidence": evidence_context,
            "entities": list(set(entities_found)),
            "latency_sec": end_time - start_time
        }

    # 5. Run Evaluation
    # LangSmith client.evaluate accepts 'data' as dataset_name (string) OR list of examples.
    # To limit, we need to fetch examples and slice them.
    if max_examples:
        print(f"Limiting to first {max_examples} examples...")
        # We need to fetch the examples from the dataset first
        limit_data = list(client.list_examples(dataset_name=dataset_name, limit=max_examples))
        experiment_results = client.evaluate(
            target,
            data=limit_data,
            evaluators=[
                health_correctness_evaluator,
                hf_statistical_evaluators,
                performance_evaluators,
                medical_grounding_evaluator,
                conciseness_evaluator,
                coherence_evaluator,
                relevance_evaluator,
                embedding_distance_evaluator
            ],
            experiment_prefix=experiment_prefix,
            max_concurrency=1,
        )
    else:
        experiment_results = client.evaluate(
            target,
            data=dataset_name,
            evaluators=[
                health_correctness_evaluator,
                hf_statistical_evaluators,
                performance_evaluators,
                medical_grounding_evaluator,
                conciseness_evaluator,
                coherence_evaluator,
                relevance_evaluator,
                embedding_distance_evaluator
            ],
            experiment_prefix=experiment_prefix,
            max_concurrency=1,
        )

    return experiment_results

if __name__ == "__main__":
    # Default behavior: Run holistic eval on default app
    results = run_evaluation(default_app, dataset_name, experiment_prefix="misinfo-workflow-holistic-eval")
    print("\nEvaluation complete!")
    print(f"View results in LangSmith: {results}")
