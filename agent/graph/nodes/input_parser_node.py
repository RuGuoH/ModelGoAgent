"""
输入解析节点
"""
from agent.config import logger, SEPARATOR
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from agent.models import GraphState

# 输入自然语言描述的项目
def input_parser_node(state: GraphState, config: RunnableConfig, *, store: BaseStore):

    return {'raw_info': state["messages"][-1].content}


# 测试版本，适用于直接输入初步报告，生成最终报告
def input_parser_node1(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """
    Stage 0：输入解析节点
    """
    # 获取用户输入
    raw_user_input = state["messages"][-1].content

    # 解析输入，用 SEPARATOR 分割
    parts = raw_user_input.split(SEPARATOR)
    if len(parts) > 1:
        structure_input = parts[0].strip()
        original_analysis = parts[1].strip()
    else:
        structure_input = raw_user_input.strip()
        original_analysis = ""

    logger.info(f"输入解析完成: structure_input 长度={len(structure_input)}, original_analysis 长度={len(original_analysis)}")

    # 返回解析后的字典
    return {
        "structure_input": structure_input,
        "original_analysis": original_analysis,
        'raw_info': state['raw_info']
    }
