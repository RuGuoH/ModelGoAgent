"""
分析节点
"""
from agent.config import logger
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from agent.models import GraphState
from agent.utils import build_stage_prompt


def analysis_node(state: GraphState, config: RunnableConfig, *, store: BaseStore, llm=None, prompt_template_system=None, prompt_template_analysis=None):
    """
    Stage 2：分析节点
    """

    original_analysis = state["original_analysis"]
    structure_input = state["structure_input"]

    # 组合成 analysis 需要的 prompt
    system_prompt = prompt_template_system.template
    user_prompt = prompt_template_analysis.template.format(
        original_analysis=original_analysis,
        structure=structure_input
    )
    prompt = build_stage_prompt(system_prompt, user_prompt)

    resp = llm.invoke(prompt)
    return {"messages": [resp]}
