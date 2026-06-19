"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _parse_verify_response(text: str) -> tuple[bool, str]:
    """Parse the JSON response from the verify node defensively."""
    # Try to find JSON in the response (model may wrap it in prose or fences)
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            ok = bool(data.get("ok", False))
            issue = str(data.get("issue", ""))
            return ok, issue
        except json.JSONDecodeError:
            pass
    # Fallback: if we can't parse JSON, assume something is wrong
    return False, f"Could not parse verifier response: {text[:200]}"


def generate_sql_node(state: AgentState) -> dict:
    """Generate SQL from the question and schema."""
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether the execution result plausibly answers the question."""
    execution_result = state.execution.render() if state.execution else "No result"

    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            execution_result=execution_result,
        )),
    ])

    ok, issue = _parse_verify_response(response.content)

    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{
            "node": "verify",
            "ok": ok,
            "issue": issue,
        }],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given the issue and the prior attempt."""
    execution_result = state.execution.render() if state.execution else "No result"

    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            execution_result=execution_result,
            issue=state.verify_issue,
        )),
    ])

    sql = _extract_sql(response.content)

    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{
            "node": "revise",
            "sql": sql,
            "issue": state.verify_issue,
        }],
    }


def route_after_verify(state: AgentState) -> str:
    """Return 'revise' to loop, 'end' to terminate."""
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()