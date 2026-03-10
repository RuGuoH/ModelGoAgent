"""
API 路由处理
"""
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from agent.config import (
    logger,
    DB_URI,
    DB_CONNECTION_KWARGS,
    DB_MAX_SIZE,
    LLM_TYPE,
    LICENSE_LLM_API_KEY,
    LICENSE_LLM_MODEL,
    GITHUB_TOKEN,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from agent.graph import create_graph
#from langgraph.checkpoint.postgres import PostgresSaver
#from langgraph.store.postgres import PostgresStore
from agent.llms import get_llm
from agent.models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, Message
from psycopg_pool import ConnectionPool
from agent.utils import format_response, save_graph_visualization
from langgraph.checkpoint.memory import MemorySaver


# 申明全局变量
graph = None
connection_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理应用生命周期：启动时初始化，关闭时清理
    """
    global graph, connection_pool
    # 启动时执行
    try:
        logger.info("正在初始化模型、定义 Graph...")

        # 配置许可证分析用 LLM（供 scripts 内 license_parser / llm_license_helper 使用）
        _scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        if LICENSE_LLM_API_KEY:
            try:
                # 始终通过模块名 `llm_license_helper` 导入，避免出现 `scripts.llm_license_helper` 与 `llm_license_helper` 两份模块
                from llm_license_helper import set_api_key
                set_api_key(api_key=LICENSE_LLM_API_KEY, model=LICENSE_LLM_MODEL, github_token=GITHUB_TOKEN or None)
                logger.info("许可证分析 LLM 已配置: model=%s", LICENSE_LLM_MODEL)
            except Exception as e:
                logger.warning("配置许可证分析 LLM 失败（未知许可证将使用保守策略）: %s", e)
        else:
            logger.debug("未设置 LICENSE_LLM_API_KEY，未知许可证将使用保守策略")

        # 初始化 LLM
        llm, embedding = get_llm(LLM_TYPE)

        # 暂时取消长期记忆

        # # 创建数据库连接池
        # connection_pool = ConnectionPool(
        #     conninfo=DB_URI,
        #     max_size=DB_MAX_SIZE,
        #     kwargs=DB_CONNECTION_KWARGS,
        # )
        # connection_pool.open()  # 显式打开连接池
        # logger.info("数据库连接池初始化成功")
        #
        # # 短期记忆 初始化checkpointer
        # checkpointer = PostgresSaver(connection_pool)
        # checkpointer.setup()
        #
        # # 长期记忆 初始化PostgresStore
        # in_postgres_store = PostgresStore(
        #     connection_pool,
        #     index={
        #         "dims": 1536,
        #         "embed": embedding
        #     }
        # )
        # in_postgres_store.setup()
        #
        # # 定义 Graph
        # graph = create_graph(llm, checkpointer, in_postgres_store)


        checkpointer = MemorySaver()
        graph = create_graph(llm, checkpointer)

        save_graph_visualization(graph)
        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        raise

    yield  # 应用运行期间

    # 关闭时执行
    logger.info("正在关闭...")
    if connection_pool:
        connection_pool.close()  # 关闭连接池
        logger.info("数据库连接池已关闭")


# 创建 FastAPI 应用
app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    封装POST请求接口，与大模型进行问答
    """
    # 判断初始化是否完成
    if not graph:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")

        query_prompt = request.messages[-1].content

        # 每次都生成新的conversationId，避免历史消息累积
        conversation_id = str(uuid.uuid4())
        user_id = request.userId if request.userId else "default_user"

        config = {"configurable": {"thread_id": user_id+"@@"+conversation_id, "user_id": user_id}}
        logger.info(f"使用新的 thread_id: {user_id}@@{conversation_id}")

        # 直接传递原始输入，由 input_parser_node 进行解析
        input_message = [
            {"role": "user", "content": query_prompt}
        ]

        # 处理流式响应
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                for message_chunk, metadata in graph.stream({"messages": input_message}, config, stream_mode="messages"):
                    chunk = message_chunk.content
                    # 在处理过程中产生每个块
                    yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {'content': chunk},'finish_reason': None}]})}\n\n"
                # 流结束的最后一块
                yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {},'finish_reason': 'stop'}]})}\n\n"
            # 返回fastapi.responses中StreamingResponse对象
            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 处理非流式响应处理
        else:
            result = ""
            try:
                events = graph.stream({"messages": input_message}, config)
                for event in events:
                    for value in event.values():
                        if "messages" in value and value["messages"]:
                            msg = value["messages"][-1]
                            # 处理字典格式和对象格式的消息
                            if isinstance(msg, dict):
                                result = msg.get("content", "")
                            else:
                                result = msg.content
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")

            formatted_response = str(format_response(result))

            response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ]
            )
            # 返回fastapi.responses中JSONResponse对象
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
