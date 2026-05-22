import os
from typing import Literal
from pydantic import BaseModel, Field

# LangChain核心组件
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入Redis历史消息类
from langchain_community.chat_message_histories import RedisChatMessageHistory

# 1.配置本地大模型
llm = ChatOllama(model="qwen3.5:9b", temperature=0.3)


# --- 1. 定义结构化输出（强制模型按这个格式返回数据）---
class TicketResult(BaseModel):
    category: Literal["物流问题", "退款问题", "商业咨询", "其他"] = Field(
        description="工单分类"
    )
    urgency: int = Field(description="紧急程度，1-5分")
    summary: str = Field(description="简要描述")


parser = PydanticOutputParser(pydantic_object=TicketResult)

# --- 2. 创建带记忆功能的提示词 ---
# MessagesPlaceholder 用来在每次对话时动态插入历史聊天记录
analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的电商客服助手。请分析用户的工单并提取关键信息。\n{format_instructions}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 4.搭建对话链（先不加 parser）
conversation_chain = analysis_prompt | llm

# 5.定义redis历史存储函数
REDIS_URL = "redis://localhost:6379/0"


def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)


# 创建一个带 Redis 消息历史的 memory_chain
memory_chain = RunnableWithMessageHistory(
    runnable=conversation_chain,
    get_session_history=get_redis_history,
    history_messages_key="chat_history",
)
# 7.运行测试
config = {"configurable": {"session_id": "user_redis_001"}}

issue1 = "我买的鞋子尺码不合适，想换一双大一号的，怎么处理？"
raw_response1 = memory_chain.invoke({"input": issue1}, config=config)
print(f"第一次回复：{raw_response1.content}")

# 你可以尝试把这段代码停止运行，然后重新跑一次
# 只要 session_id 还是 "user_redis_001"，Redis 里的历史对话依然存在！
issue2 = "另外，换货的运费是谁承担呢？"
raw_response2 = memory_chain.invoke({"input": issue2}, config=config)
print(f"第二次回复：{raw_response2.content}")

# 手动解析最终结果（保持和之前一致）
result = parser.parse(raw_response2.content)
print(f"结构化结果：{result.model_dump_json(indent=2, ensure_ascii=False)}")
