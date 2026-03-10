"""
图构建逻辑
"""
from functools import partial

from agent.config import (
    PROMPT_TEMPLATE_TXT_SYS,
    PROMPT_TEMPLATE_TXT_STR,
    PROMPT_TEMPLATE_TXT_ANA,
    PROMPT_TEMPLATE_TXT_WORK,
    PROMPT_TEMPLATE_TXT_REUSE,
    PROMPT_TEMPLATE_TXT_CODE,
    PROMPT_TEMPLATE_TXT_POLICY,
    PROMPT_TEMPLATE_TXT_REUSE_AMEND,
    logger
)
from agent.graph.nodes import input_parser_node1, work_identifier_node, analysis_node, input_parser_node, \
    reuse_method_node, reuse_method_amend_node, generate_code, release_policy_node
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from agent.models import GraphState


def create_graph(llm, checkpointer, in_postgres_store=None) -> StateGraph:
    """
    创建和配置 chatbot 的状态图
    """
    try:
        # 构建 graph，使用自定义 GraphState
        graph_builder = StateGraph(GraphState)

        # ------- 加载 prompt 模板 -------
        prompt_template_system = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_SYS, encoding="utf-8")
        prompt_template_structure = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_STR, encoding="utf-8")
        prompt_template_analysis = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_ANA, encoding="utf-8")
        prompt_template_work = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_WORK, encoding="utf-8")
        prompt_template_reuse = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_REUSE, encoding="utf-8")
        prompt_template_reuse_amend = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_REUSE_AMEND, encoding="utf-8")
        prompt_template_code = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_CODE, encoding="utf-8")
        prompt_template_policy = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_POLICY, encoding="utf-8")


        # ===== 注册节点 =====
        # 使用 partial 绑定额外的参数
        # graph_builder.add_node("input_parser_node", input_parser_node)
        graph_builder.add_node('input_parser_node',input_parser_node)
        graph_builder.add_node(
            "release_policy_node",
            partial(release_policy_node, llm=llm, prompt_template_work=prompt_template_policy)
        )
        graph_builder.add_node(
            "work_identifier_node",
            partial(work_identifier_node, llm=llm, prompt_template_work=prompt_template_work)
        )
        graph_builder.add_node(
            "reuse_node",
            partial(reuse_method_node, llm=llm, prompt_template_work=prompt_template_reuse)
        )
        graph_builder.add_node(
            "reuse_amend_node",
            partial(reuse_method_amend_node, llm=llm, prompt_template_work=prompt_template_reuse_amend)
        )
        graph_builder.add_node(
            "code_node",
            partial(generate_code, llm=llm, prompt_template_work=prompt_template_code)
        )
        graph_builder.add_node(
            "analysis_node",
            partial(analysis_node, llm=llm, prompt_template_system=prompt_template_system, prompt_template_analysis=prompt_template_analysis)
        )

        # ===== 配置边（chain）=====
        graph_builder.add_edge(START, "input_parser_node")
        graph_builder.add_edge("input_parser_node", "release_policy_node")
        graph_builder.add_edge("release_policy_node", "work_identifier_node")
        graph_builder.add_edge('work_identifier_node', "reuse_node")
        graph_builder.add_edge('reuse_node', "reuse_amend_node")
        graph_builder.add_edge('reuse_amend_node', "code_node")
        graph_builder.add_edge('code_node', 'analysis_node')
        graph_builder.add_edge("analysis_node", END)

        graph_builder.add_edge('code_node', END)

        # 编译生成 graph 并返回
        return graph_builder.compile(checkpointer=checkpointer, store=in_postgres_store)

    except Exception as e:
        raise RuntimeError(f"创建 graph 失败: {str(e)}")
