from deepagents import create_deep_agent, CompiledSubAgent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Import our custom modules
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
    debate_adjudicator_tool
)

# Load environment variables
load_dotenv()

# --- A2: Evidence Retrieval Nested Workflow ---

A2_SYSTEM_PROMPT = """You are the Evidence Retrieval Orchestrator. 
Your goal is to conduct deep research by coordinating specialized agents.
Decompose the claim, search diversified sources (Web + PubMed), summarize, audit for gaps, and synthesize a final report.
CRITICAL: You MUST end your final response with 'RESULT_SUMMARY: <detailed research findings>'.
"""

a2_subagents = [
    {
        "name": "sub-questions-planner",
        "description": "Decomposes the main question into 3–7 focused sub-questions and cross-lingual variants.",
        "system_prompt": "You are a research planner. Use the cross_lingual_bundle_tool to expand the search scope.",
        "tools": [cross_lingual_bundle_tool],
        "model": "gpt-4o"
    },
    {
        "name": "research-assistant",
        "description": "Issues diversified queries to general web search and PubMed.",
        "system_prompt": "You are a research assistant. Use evidence_retrieval_tool and pubmed_search_tool. Apply filtering_and_dedup_tool to clean results.",
        "tools": [evidence_retrieval_tool, pubmed_search_tool, filtering_and_dedup_tool],
        "model": "gpt-4o"
    },
    {
        "name": "summarization-expert",
        "description": "Extracts 2–4 claim-relevant bullets per source.",
        "system_prompt": "You are a source compressor. Extract concise, relevant facts from research findings.",
        "tools": [],
        "model": "gpt-4o"
    },
    {
        "name": "research-reviewer",
        "description": "Audits coverage and flags gaps.",
        "system_prompt": "You are a research auditor. Ensure all sub-questions are answered and flag missing populations or data.",
        "tools": [],
        "model": "gpt-4o"
    },
    {
        "name": "research-writer",
        "description": "Produces a concise, structured report with inline citations.",
        "system_prompt": "You are a synthesizer. Produce a themed report with citations based on the audited evidence.",
        "tools": [],
        "model": "gpt-4o"
    }
]

# Create the nested A2 agent
a2_compiled_agent = create_deep_agent(
    subagents=a2_subagents,
    system_prompt=A2_SYSTEM_PROMPT,
    model="gpt-4o",
    name="evidence-retrieval-depth"
)

# --- A3: Reasoning & Explanation Nested Workflow ---

A3_SYSTEM_PROMPT = """You are the Reasoning and Explanation Orchestrator.
Your goal is to synthesize claims and evidence into a transparent, auditable judgment.
Coordinate your sub-agents in a strict sequence:
1. Prioritize evidence.
2. Extract verbatim grounding spans.
3. Assign veracity labels and compute calibrated confidence.
4. Compose the final neutral explanation.

Use the `task(name, task)` tool.
CRITICAL: You MUST end your final response with 'RESULT_SUMMARY: Label=<label>, Confidence=<score>, Explanation=<text>'.
"""

a3_subagents = [
    {
        "name": "prioritizer",
        "description": "Ranks and prunes evidence based on source credibility and recency.",
        "system_prompt": "You are an evidence prioritizer. Use the evidence_prioritization_tool to rank retrieved passages.",
        "tools": [evidence_prioritization_tool],
        "model": "gpt-4o"
    },
    {
        "name": "span-analyst",
        "description": "Extracts verbatim spans and identifies support/counter signals.",
        "system_prompt": "You are a span analyst. Use verbatim_span_extraction_tool to ground every reasoning step in explicit text.",
        "tools": [verbatim_span_extraction_tool],
        "model": "gpt-4o"
    },
    {
        "name": "calibrator",
        "description": "Assigns veracity labels and computes calibrated confidence scores.",
        "system_prompt": "You are a veracity calibrator. Use the logistic_calibration_tool to assign labels and compute high-integrity confidence scores.",
        "tools": [logistic_calibration_tool, reasoning_explanation_tool],
        "model": "gpt-4o"
    },
    {
        "name": "explanation-summarizer",
        "description": "Composes concise, grounded explanations and follow-up queries.",
        "system_prompt": """You are an explanation specialist. 
1. Synthesize the most informative spans into a neutral, user-facing explanation (<= 120 words).
2. CRITICAL: Your final output MUST end with 'RESULT_SUMMARY: Label=<label>, Confidence=<score>, Explanation=<text>'.
""",
        "tools": [],
        "model": "gpt-4o"
    }
]

# Create the nested A3 agent
a3_compiled_agent = create_deep_agent(
    subagents=a3_subagents,
    system_prompt=A3_SYSTEM_PROMPT,
    model="gpt-4o",
    name="reasoning-explanation-depth"
)

# --- A4: Critic & Debate Nested Workflow ---

A4_SYSTEM_PROMPT = """You are the Critic & Debate Orchestrator.
Your goal is to verify the veracity judgment from A3 before reporting.
Detect hallucinations, enforce recency/credibility, and manage debates for conflicting signals.
CRITICAL: You MUST end your final response with 'RESULT_SUMMARY: Status=<status>, Flags=<flags>, Adjusted_Confidence=<score>'.
"""

a4_subagents = [
    {
        "name": "auditor",
        "description": "Validates schema integrity and span faithfulness.",
        "system_prompt": "You are a faithfulness auditor. Use span_faithfulness_tool to ensure all quotes exactly match source passages.",
        "tools": [span_faithfulness_tool],
        "model": "gpt-4o"
    },
    {
        "name": "policy-enforcer",
        "description": "Enforces recency, credibility, and sanity constraints.",
        "system_prompt": "You are a policy enforcer. Use recency_decay_tool to flag stale guidance and ensure minimum high-credibility source counts.",
        "tools": [recency_decay_tool],
        "model": "gpt-4o"
    },
    {
        "name": "debater",
        "description": "Manages pro/con rounds for conflicting signals.",
        "system_prompt": "You are a debate facilitator. Use debate_adjudicator_tool to resolve conflicts between supporting and countering evidence.",
        "tools": [debate_adjudicator_tool],
        "model": "gpt-4o"
    },
    {
        "name": "gatekeeper",
        "description": "Applies final confidence adjustments and routes to reporting.",
        "system_prompt": """You are a governance gatekeeper. 
1. Use confidence_adjustment_tool to apply penalties and finalize the decision package.
2. CRITICAL: Your final output MUST end with 'RESULT_SUMMARY: Status=<status>, Flags=<flags>, Adjusted_Confidence=<score>'.
""",
        "tools": [confidence_adjustment_tool],
        "model": "gpt-4o"
    }
]

# Create the nested A4 agent
a4_compiled_agent = create_deep_agent(
    subagents=a4_subagents,
    system_prompt=A4_SYSTEM_PROMPT,
    model="gpt-4o",
    name="critic-debate-depth"
)

# --- Main Orchestration ---

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Master Orchestrator for health misinformation verification.
MANDATORY PIPELINE - YOU MUST DELEGATE IN THIS EXACT ORDER:

1. `claim-parsing`
2. `evidence-retrieval`
3. `reasoning-explanation`
4. `critic-calibration`
5. `final-reporting`

YOUR FINAL RESPONSE MUST BE THE RAW JSON RETURNED BY THE `final-reporting` AGENT.
DO NOT ADD ANY OTHER TEXT. NO INTROS. NO OUTROS. NO MARKDOWN.
IF YOU OUTPUT ANYTHING OTHER THAN RAW JSON, THE SYSTEM WILL FAIL.
"""

subagents = [
    {
        "name": "claim-parsing",
        "description": "Decomposes input text into atomic, verifiable health claims.",
        "system_prompt": "Use the claim_parsing_tool to process input.",
        "tools": [claim_parsing_tool],
        "model": "gpt-4o"
    },
    CompiledSubAgent(
        name="evidence-retrieval",
        description="Nested 5-agent research workflow for deep evidence gathering (Web + PubMed + Multi-lingual).",
        runnable=a2_compiled_agent
    ),
    CompiledSubAgent(
        name="reasoning-explanation",
        description="Nested 4-agent reasoning workflow for grounded veracity assessment and calibrated confidence.",
        runnable=a3_compiled_agent
    ),
    CompiledSubAgent(
        name="critic-calibration",
        description="Nested 4-agent governance workflow for faithfulness auditing, policy enforcement, and debate.",
        runnable=a4_compiled_agent
    ),
    {
        "name": "final-reporting",
        "description": "Synthesizes final JSON report.",
        "system_prompt": """SYSTEM PROTOCOL:
1. Scan the history for all 'RESULT_SUMMARY' blocks.
2. Extract labels, evidence, confidence, and explanations.
3. Call `final_reporting_tool(approved_results=...)` with a list of dictionaries.
4. YOUR FINAL OUTPUT MUST BE THE EXACT JSON STRING RETURNED BY THE TOOL. NO OTHER TEXT. NO MARKDOWN.""",
        "tools": [final_reporting_tool],
        "model": "gpt-4o"
    }
]

# Create the master agent
app = create_deep_agent(
    subagents=subagents,
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    model="gpt-4o"
)

if __name__ == "__main__":
    print("--- Orchestrator with Nested A2, A3, & A4 Workflows Initialized ---")
    user_input = "Vaccines can cause autism."
    print(f"User Input: {user_input}\n")
    
    try:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print("\n--- Final Report ---")
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()

