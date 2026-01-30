from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    input_text: str
    claims: Optional[List[Dict[str, Any]]]
    evidence: Optional[Dict[str, Any]]
    reasoning: Optional[Dict[str, Any]]
    reviews: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]
    next_agent: str
