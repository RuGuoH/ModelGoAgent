"""
配置和常量定义
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# 设置LangSmith环境变量 进行应用跟踪，实时了解应用中的每一步发生了什么
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# prompt模版设置相关 根据自己的实际业务进行调整
PROMPT_TEMPLATE_TXT_SYS = "./prompt/prompt_template_system.txt"
PROMPT_TEMPLATE_TXT_STR = "./prompt/prompt_template_structure.txt"
PROMPT_TEMPLATE_TXT_ANA = "./prompt/prompt_template_analysis.txt"
PROMPT_TEMPLATE_TXT_WORK = "./prompt/prompt_template_work.txt"
PROMPT_TEMPLATE_TXT_REUSE = './prompt/prompt_template_reuse.txt'
PROMPT_TEMPLATE_TXT_CODE = "./prompt/prompt_template_code.txt"
PROMPT_TEMPLATE_TXT_POLICY = "./prompt/prompt_template_open_policy&type.txt"
PROMPT_TEMPLATE_TXT_REUSE_AMEND = "./prompt/prompt_template_reuse_amend.txt"

# 分隔符
SEPARATOR = "---------------------------------------------------------------------------------------------"

# openai:调用gpt模型,oneapi:调用oneapi方案支持的模型,ollama:调用本地开源大模型,qwen:调用阿里通义千问大模型
LLM_TYPE = "openai"

# API服务设置相关
PORT = 8012

# 数据库配置
DB_URI = "postgresql://postgres:postgres@localhost:5433/postgres?sslmode=disable"
DB_CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
}
DB_MAX_SIZE = 20

E2B_API_KEY = os.getenv("E2B_API_KEY", "")

# 许可证分析用 LLM（未知许可证识别与建模）
LICENSE_LLM_API_KEY = os.getenv("LICENSE_LLM_API_KEY", "")
LICENSE_LLM_MODEL = os.getenv("LICENSE_LLM_MODEL", "deepseek")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
