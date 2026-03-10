"""
代码生成与本地执行节点
"""
import io
import contextlib

from agent.config import logger, SEPARATOR
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from agent.models import GraphState
from agent.utils import build_stage_prompt
from .helpers import extract_multiple_functions, extract_python_code


def generate_code(state: GraphState, config: RunnableConfig, *, store: BaseStore, llm=None, prompt_template_work=None):
    """
    基于复用方法描述生成 ModelGo 分析代码，并在本地直接执行，得到：
    - original_analysis：规则引擎输出的详细分析文本
    - structure_input：最终 Work 结构的简要表示
    """
    known_works = state["known_works"]
    raw_info = state["raw_info"]
    reuse_method = state["reuse_method"]
    open_policy = state["open_policy"]
    open_type = state["open_type"]

    logger.info(f"复用方法解析结果: {reuse_method}")
    reuse_method_name = [i["method"] for i in reuse_method]

    # 让 LLM 根据复用方法 JSON 生成可执行的 Python 代码片段
    user_prompt = prompt_template_work.template.format(
        description=raw_info,
        known_work_dict=known_works,
        reuse_method=reuse_method,
        reuse_code=extract_multiple_functions(reuse_method_name),
    )
    prompt = build_stage_prompt("", user_prompt)
    resp = llm.invoke(prompt)
    context = resp.content

    # 组装完整可执行脚本
    code_list = ["from main_case import *"]
    for work in known_works:
        if work.code:
            code_list.append(work.code)

    code_list.append("works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]")
    code_list.append("par.register_license(works)")
    code_list.append(extract_python_code(context))
    code_list.append(f"new_work.form = '{open_type}'")
    code_list.append(f"par.analysis(new_work, open_policy='{open_policy}')")
    code_list.append("new_work.summary()")
    code_list.append(f"print('\\n{SEPARATOR}\\n')")
    code_list.append("print(new_work)")

    code = "\n".join(code_list)
    logger.info(f"生成的分析代码:\n{code}")

    # 在当前进程中本地执行代码，并捕获输出
    stdout_buffer = io.StringIO()
    try:
        exec_globals = {}
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, exec_globals)
    except Exception as e:
        logger.error(f"本地执行生成代码失败: {e}")
        raise

    stdout = stdout_buffer.getvalue()
    parts = stdout.split(SEPARATOR)

    if len(parts) >= 2:
        original_analysis = parts[0]
        structure_input = parts[1]
    else:
        logger.error("执行结果未包含预期分隔符，无法拆分 original_analysis 与 structure_input")
        original_analysis = stdout
        structure_input = ""

    return GraphState(
        messages=state["messages"] + [resp],
        original_analysis=original_analysis.strip(),
        structure_input=structure_input.strip(),
    )
