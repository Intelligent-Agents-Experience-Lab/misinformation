# Experimentation and Results

This section details the comprehensive evaluation of the Health Misinformation Verification Workflow. We conducted a rigorous set of experiments including a baseline assessment, ablation studies to isolate component contributions, and a model benchmark to evaluate performance across different LLM backends.

## 1. Experimental Setup

### 1.1 Configuration
The core evaluation was performed on the **Baseline** configuration, designed to represent the full capabilities of the system:
*   **Orchestrator**: GPT-4o
*   **Tools**: Claim Parsing, Evidence Retrieval (Google Search), PubMed Search, Reasoning, Critic, Final Reporting.
*   **Advanced Features**: UMLS Entity Normalization, Evidence Filtering, Span Extraction, Confidence Calibration, and Debate Protocols.

### 1.2 Dataset
We utilized the `Health_Misinformation_Eval_Subset`, a curated dataset of health-related claims labeled as "True" or "Misinformation". This dataset challenges the model with nuanced medical claims requiring external verification.

### 1.3 Evaluation Framework
Experiments were managed via **LangSmith** using a custom evaluation suite:
1.  **Correctness & Grounding**:
    *   *Correctness*: Alignment with ground truth labels.
    *   *Medical Grounding*: Support of reasoning by retrieved evidence (crucial for hallucination prevention).
2.  **Agentic Performance**:
    *   *Plan Adherence*: Fidelity to the orchestrator's stated plan.
    *   *Tool Correctness*: Appropriateness of tool selection and sequencing.
3.  **Efficiency**:
    *   *Latency*: End-to-end execution time.
    *   *Step Efficiency*: Avoidance of redundant steps.

## 2. Experimental Results

### 2.1 Baseline Performance
The Baseline configuration (GPT-4o) establishes the standard for high-fidelity verification.

| Metric Category | Metric Name | Score / Value |
| :--- | :--- | :--- |
| **Correctness** | **Medical Grounding** | **0.84** |
| | Relevance | 0.81 |
| | Task Completion | 0.77 |
| | Coherence | 0.75 |
| | Correctness (Verdict) | 0.47 |
| **Agentic** | **Plan Adherence** | **0.82** |
| | Plan Quality | 0.70 |
| | Tool Correctness | 0.59 |
| **Performance** | Latency (mean) | 18.79s |

*Key Insight*: The system excels at **Medical Grounding (0.84)** and **Plan Adherence (0.82)**, indicating it is a "safe" agent that sticks to instructions and rarely hallucinates unsupported claims. However, the absolute **Correctness (0.47)** of the binary True/False verdict suggests room for improvement in final decision boundaries.

### 2.2 Ablation Study
To understand the contribution of each architectural component, we evaluated several variants against the baseline.

| Experiment Configuration | Med. Grounding | Correctness | Plan Adherence | Latency (s) | Impact Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** (All features) | 0.84 | 0.47 | **0.82** | 18.79 | Balanced performance. |
| **CoT Orchestrator** | 0.85 | **0.52** | 0.80 | 22.45 | **Chain-of-Thought** improves correctness (+5%) but increases latency. |
| **Med Persona** | 0.83 | 0.49 | 0.81 | 19.10 | Medical persona slightly aids correctness. |
| **PubMed Only** | **0.88** | 0.42 | 0.79 | 15.30 | Highest grounding (+4%), but misses non-academic context. |
| **Web Only** | 0.75 | 0.45 | 0.78 | 14.80 | Faster, but significantly lowers grounding (-9%). |
| **No Calibration** | 0.83 | 0.41 | 0.82 | 16.50 | removing critic lowers correctness (-6%). |
| **No Debate** | 0.80 | 0.44 | 0.81 | **12.20** | Fastest (-35% latency), minor accuracy drop. |
| **Zero Shot** (No Tools) | 0.15 | 0.32 | N/A | 3.50 | Complete failure on grounding; effectively guessing. |

*Key Insight*: The **PubMed Only** variant achieves the highest grounding score, confirming the value of specialized tools. **CoT Orchestrator** provides the best correctness trade-off, justifying its computational cost.

### 2.3 Model Benchmarks
Evaluation of the workflow across different LLM backends to assess cost-performance trade-offs.

| Model | Provider | Correctness | Med. Grounding | Cost / 1k Runs (Est) | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-4o** | OpenAI | 0.47 | **0.84** | High | **Best for Production** (High safety). |
| **Claude 3.5 Sonnet** | OpenRouter | **0.49** | 0.83 | High | Top correctness, excellent alternative. |
| **DeepSeek-R1** | OpenRouter | 0.46 | 0.82 | Medium | Strong open-weights contender. |
| **Llama-3.3-70b** | OpenRouter | 0.45 | 0.81 | Medium | Reliable, good balance. |
| **GPT-4o-mini** | OpenAI | 0.41 | 0.78 | Low | Best budget option. |
| **Mistral Small 24B** | OpenRouter | 0.39 | 0.72 | Low | Acceptable for low-risk screening. |
| **Gemma 2 27B** | OpenRouter | 0.38 | 0.70 | Low | High hallucination risk. |

## 3. Discussion

### 3.1 Architectural Insights
The ablation study reveals that **Tool Selection is critical for Grounding**. The "Web Only" configuration suffered a nearly 10-point drop in Medical Grounding compared to the Baseline, underscoring the necessity of the `pubmed_search_tool` for validating health claims. Conversely, while "PubMed Only" excelled at grounding, its drop in Correctness (0.42) suggests that many "Misinformation" claims originate from popular culture or social media contexts that are not indexed in medical literature, requiring the broad net of web search to debunk effectively.

The **Chain-of-Thought (CoT)** prompt variant yielded the highest Correctness (0.52). By forcing the model to explicitly step through its reasoning before committing to a tool or a verdict, we reduce premature convergence on incorrect answers. This comes at a latency cost (22.45s vs 18.79s), which is a worthwhile trade-off for high-stakes health verification.

### 3.2 Evaluation of Open Source Models
A key finding is the competitiveness of **DeepSeek-R1** and **Llama-3.3-70b**. DeepSeek-R1 achieved a Medical Grounding score (0.82) nearly matching GPT-4o (0.84) and a Correctness score (0.46) comparable to the baseline. This suggests that for organizations with data privacy constraints requiring self-hosted models, open-weights models are now a viable alternative to proprietary APIs for complex agentic workflows.

However, smaller models like **Gemma 2 27B** and **Mistral Small** struggled significantly with instruction following in the multi-agent context, often failing to call the correct tools or hallucinating arguments, as evidenced by lower Grounding scores (<0.72).

### 3.3 Challenges and Limitations
1.  **Metric Strictness**: The `Correctness` score (0.47) is notably lower than qualitative observation would suggest. Analysis shows this is partly due to the binary nature of the metric. If the model answers "Mostly True" with caveats, and the ground truth is "True", it may be penalized.
2.  **Latency**: With an average latency of ~19 seconds, the real-time application of this workflow in a user-facing chat interface is challenging. The "No Debate" ablation shows we can cut this to ~12 seconds, but users needing instant feedback may still find this too slow.
3.  **Tool Ordering**: The "Tool Correctness" score of 0.59 indicates the orchestrator sometimes struggles with optimal sequencing (e.g., searching before parsing). Few-shot prompting has mitigated this, but further fine-tuning may be required.

## 4. Conclusion and Future Work
This study demonstrates that an agentic workflow with specialized medical tools significantly outperforms zero-shot LLM predictions in verifying health misinformation. The **Baseline** configuration provides a robust balance of safety (Grounding) and accuracy.

**Future directions include**:
*   **Hybrid Routing**: Using a small, fast model (e.g., GPT-4o-mini) to route easy claims and reserving GPT-4o/Claude 3.5 for complex medical reasoning.
*   **Prompt Optimization**: Refining the system prompt to improve Tool Correctness scores.
*   **Streaming UI**: Implementing streaming responses to mitigate the perceived latency of the 19-second verification loop.
