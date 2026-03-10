"""
复用方法节点
"""
from agent.config import logger
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from agent.models import GraphState
from agent.utils import build_stage_prompt
from .helpers import safe_json_loads, validate_reuse_method_resp


def reuse_method_node(state: GraphState, config: RunnableConfig, *, store: BaseStore,llm=None, prompt_template_work=None):
    known_works = state["known_works"]
    raw_info = state["raw_info"]
    # str(list(known_works.keys()))
    user_prompt = prompt_template_work.template.format(description = raw_info, works = [i.standard_name for i in known_works])
    prompt = build_stage_prompt('', user_prompt)
    resp = llm.invoke(prompt)
    result = safe_json_loads(resp.content)
    print(f'第一次reuse输出：{result}')

    if not validate_reuse_method_resp(result):
        logger.warning("[格式错误] LLM 输出不符合规范")
    else:
        logger.warning("reuse method输出格式符合要求")

    return {
        "raw_info": state['raw_info'],
        "messages": state["messages"],
        "known_works": state["known_works"],
        "unknown_works": state["unknown_works"],
        "reuse_method": result
    }
