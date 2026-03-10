"""
图节点函数定义
"""
import ast
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

from agent.config import logger, SEPARATOR
from agent.knowledge import registered_work
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from agent.models import GraphState, Work
from agent.utils import build_stage_prompt
from e2b_code_interpreter import Sandbox

def input_parser_node(state: GraphState, config: RunnableConfig, *, store: BaseStore):

    return {'raw_info': state["messages"][-1].content}

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

def release_policy_node(state: GraphState, config: RunnableConfig, *, store: BaseStore, llm=None, prompt_template_work=None):
    user_input = state['raw_info']
    prompt = build_stage_prompt('', prompt_template_work.template.format(input=user_input))
    resp = llm.invoke(prompt)
    try:
        result = json.loads(resp.content)
        open_policy = result['open_policy']
        open_type = result['open_type']

        logger.info(f'new work的发布方式为{open_policy}，发布形式为{open_type}')
        return GraphState(
            messages=[{"role": "assistant", "content": f'new work的发布方式为{open_policy}，发布形式为{open_type}'}],
            open_policy=open_policy,
            open_type=open_type
        )
    except Exception as e:
        logger.error(e)



def work_identifier_node(state: GraphState, config: RunnableConfig, *, store: BaseStore, llm=None, prompt_template_work=None):
    """
    Stage 0.5：Work 识别节点
    """
    # 获取用户输入
    user_input = state['raw_info']
    prompt = build_stage_prompt('', prompt_template_work.template.format(description=user_input, known_works=str(list(registered_work.keys()))))
    resp = llm.invoke(prompt)

    identified_works_list = []
    unknown_works_list = []

    # 解析 LLM 返回的结果
    try:
        result = json.loads(resp.content)

        for item in result:
            k = list(item.keys())[0]
            v = item[k]
            if v in registered_work.keys():
                identified_works_list.append(Work(k, v, registered_work[v]))
            else:
                unknown_works_list.append(Work(k, v))

        logger.info(f"识别出的已知 Work:{identified_works_list}")
        logger.info(f"未知对象列表: {unknown_works_list}")

        if len(unknown_works_list) > 0:
            error_msg = (
                "检测到未识别的 Work，对话已中断。\n"
                f"未知 Work: {unknown_works_list}\n"
                "请先注册这些 Work 或修正输入后重试。"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 返回结果并添加到消息中
        result_message = f"已识别的 Work 对象:\n{[i.name for i in identified_works_list]}\n\n未知对象列表: {unknown_works_list}"

    except:
        logger.warning(f"无法解析 LLM 返回的 JSON: {resp.content}")

    return {
        "raw_info": state['raw_info'],
        "messages": [{"role": "assistant", "content": result_message}],
        "known_works": identified_works_list,
        "unknown_works": unknown_works_list
    }

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

def reuse_method_amend_node(state: GraphState, config: RunnableConfig, *, store: BaseStore,llm=None, prompt_template_work=None):
    known_works = state["known_works"]
    raw_info = state["raw_info"]
    reuse_method = state["reuse_method"]
    # str(list(known_works.keys()))
    user_prompt = prompt_template_work.template.format(
        description = raw_info,
        works = [i.standard_name for i in known_works],
        reuse_method = reuse_method,
        reuse_code=extract_multiple_functions([i['method'] for i in reuse_method])
    )
    prompt = build_stage_prompt('', user_prompt)
    resp = llm.invoke(prompt)
    result = safe_json_loads(resp.content)
    print(f'第二次reuse输出：{result}')

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

def generate_code(state: GraphState, config: RunnableConfig, *, store: BaseStore,llm=None, prompt_template_work=None):
    known_works = state["known_works"]
    raw_info = state["raw_info"]
    reuse_method = state["reuse_method"]
    open_policy = state["open_policy"]
    open_type = state["open_type"]

    logger.info(reuse_method)
    reuse_method_name = [i['method'] for i in reuse_method]

    user_prompt = prompt_template_work.template.format(
        description = raw_info,
        known_work_dict = known_works,
        reuse_method= reuse_method,
        reuse_code=extract_multiple_functions(reuse_method_name))
    prompt = build_stage_prompt('', user_prompt)
    resp = llm.invoke(prompt)
    context = resp.content
    code_list = ['from main_case import *']
    for work in known_works:
        code_list.append(work.code)

    code_list.append("works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]")
    code_list.append("par.register_license(works)")

    code_list.append(extract_python_code(context))
    # code_list.append(context.replace('```', ''))

    code_list.append(f"new_work.form = '{open_type}'")

    code_list.append(f"par.analysis(new_work, open_policy='{open_policy}')")
    code_list.append("new_work.summary()")
    code_list.append(f"print('\\n{SEPARATOR}\\n')")
    code_list.append("print(new_work)")

    code = "\n".join(code_list)
    print(f'生成的代码为\n{code}')

    # 在 E2B 沙盒中运行代码
    try:
        with Sandbox.create('model-go') as sbx:
            sbx.files.write('/home/user/scripts/tmp.py', code)
            result = sbx.commands.run(
                "cd /home/user/scripts && python tmp.py"
            )
            stdout = result.stdout

        parts = stdout.split(SEPARATOR)

        original_analysis = parts[0]
        structure_input = parts[1]

    except Exception as e:
        logger.error(f'E2B 沙盒执行失败: {e}')

    return GraphState(
        messages = state["messages"] + [resp],
        original_analysis = original_analysis,
        structure_input = structure_input,
    )


def structure_node(state: GraphState, config: RunnableConfig, *, store: BaseStore, llm=None, prompt_template_system=None, prompt_template_structure=None):
    logger.info("111111111111111111111")
    structure_input = state["structure_input"]
    logger.info(f"structure_input = {structure_input}")

    system_prompt = prompt_template_system.template
    user_prompt = prompt_template_structure.template.format(new_work=structure_input)

    prompt = build_stage_prompt(system_prompt, user_prompt)

    resp = llm.invoke(prompt)

    # 将 structure 内容保存，供下一个节点使用
    return {
        "messages": [resp],
    }


def analysis_node(state: GraphState, config: RunnableConfig, *, store: BaseStore, llm=None, prompt_template_system=None, prompt_template_analysis=None):
    """
    Stage 2：分析节点
    """
    # structure 节点输出内容
    structured_output = state["messages"][-1].content
    logger.info(f'新作品的结构为：\n{structured_output}')

    # 从 state 获取 original_analysis
    original_analysis = state["original_analysis"]

    # 组合成 analysis 需要的 prompt
    system_prompt = prompt_template_system.template
    user_prompt = prompt_template_analysis.template.format(
        original_analysis=original_analysis,
        structure=structured_output
    )
    prompt = build_stage_prompt(system_prompt, user_prompt)

    resp = llm.invoke(prompt)
    return {"messages": [resp]}

def validate_reuse_method_resp(data) -> bool:
    """
    简化版校验 LLM 输出是否符合预期格式：
    [
      {
        "method": "",
        "description": "",
        "inputs": [],
        "output": ""
      }
    ]
    返回 True/False
    """
    try:
        if not isinstance(data, list):
            return False

        for item in data:
            if not isinstance(item, dict):
                return False
            if not all(key in item for key in ["method", "description", "inputs", "output"]):
                return False
            if not isinstance(item["inputs"], list):
                return False

        return True
    except Exception:
        return False

def safe_json_loads(text: str):
    """
    安全解析 LLM 输出的 JSON。
    - 自动提取 ```json``` codeblock
    - 自动提取第一个 JSON 块
    - 清洗不可见字符
    - 失败时返回 None，并打印 logger.info 错误原因
    """

    try:
        raw = text.strip()

        # 提取 ```json ... ```
        code_block = re.search(r"```json(.*?)```", raw, re.S)
        if code_block:
            raw = code_block.group(1).strip()

        # 提取第一个 JSON 块（数组或对象）
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
        if match:
            raw = match.group(1)

        # 清洗 Unicode 不可见字符
        raw = raw.encode("utf-8", "ignore").decode("utf-8")

        return json.loads(raw)

    except Exception as e:
        logger.error(f"safe_json_loads 解析失败: {e}")
        return None

# if __name__ == "__main__":
#     result = [{'method': 'embed', 'description': '将输入的工作（deep-sequoia 和 pubmed）转换为数值向量或表征，适用于翻译场景。', 'inputs': ['deep-sequoia', 'pubmed'], 'output': '翻译后的deep-sequoia和pubmed'}, {'method': 'combine', 'description': '将翻译后的 deep-sequoia 和 pubmed 与 bigtranslate 并列组合成新的组合型工作，多个组件可以分离。', 'inputs': ['翻译后的deep-sequoia', '翻译后的pubmed', 'bigtranslate'], 'output': '组合型工作'}]
#     print(validate_reuse_method_resp(result))


def extract_function_from_file(function_name: str, file_path: str = None) -> str:
    """
    从指定的 Python 文件中提取函数定义及其文档字符串

    Args:
        function_name: 要提取的函数名
        file_path: Python 文件路径，默认为 ../scripts/reuse_methods.py

    Returns:
        包含函数定义的文本字符串，如果未找到则返回空字符串
    """
    if file_path is None:
        # 默认路径：相对于当前文件的位置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "..", "..", "scripts", "reuse_methods.py")

    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            file_lines = file_content.split('\n')

        # 解析 AST
        tree = ast.parse(file_content)

        # 查找目标函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # 获取函数的起始行和结束行
                start_line = node.lineno - 1  # AST 行号从 1 开始
                end_line = node.end_lineno

                # 提取函数前的注释（如果有）
                comment_lines = []
                check_line = start_line - 1
                while check_line >= 0:
                    line = file_lines[check_line].strip()
                    # 检查是否是注释或文档字符串
                    if line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
                        comment_lines.insert(0, file_lines[check_line])
                        check_line -= 1
                    elif line == '':
                        # 空行也包含
                        comment_lines.insert(0, file_lines[check_line])
                        check_line -= 1
                    else:
                        break

                # 提取函数代码
                function_code = '\n'.join(file_lines[start_line:end_line])

                # 组合注释和函数代码
                if comment_lines:
                    result = '\n'.join(comment_lines) + '\n' + function_code
                else:
                    result = function_code

                logger.info(f"成功提取函数 '{function_name}' ({end_line - start_line} 行)")
                return result

        logger.warning(f"未找到函数 '{function_name}' 在文件 {file_path}")
        return ""

    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        return ""
    except SyntaxError as e:
        logger.error(f"文件语法错误: {e}")
        return ""
    except Exception as e:
        logger.error(f"提取函数时出错: {e}")
        return ""


def extract_multiple_functions(function_names: list, file_path: str = None) -> str:
    """
    从指定的 Python 文件中提取多个函数定义

    Args:
        function_names: 要提取的函数名列表
        file_path: Python 文件路径，默认为 ../scripts/reuse_methods.py

    Returns:
        包含所有函数定义的文本字符串，用分隔符分开
    """
    results = []

    function_names = list(set(function_names))
    for func_name in function_names:
        func_code = extract_function_from_file(func_name, file_path)
        if func_code:
            results.append(func_code)

    return "\n\n".join(results) if results else ""

def extract_python_code(text: str) -> str:
    m = re.search(r'```python\s*(.*?)\s*```', text, re.S | re.I)
    if m:
        return m.group(1)

    m = re.search(r'```\s*(.*?)\s*```', text, re.S)
    if m:
        return m.group(1)

    return text.strip()
