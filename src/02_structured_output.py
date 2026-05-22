import os
from typing import Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 初始化大模型
llm = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434", temperature=0.7)


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

# --- 3. 搭建带记忆的对话链 ---
# 将提示词、模型和解析器串联
chain = analysis_prompt | llm

#########################################################
# 使用内存来存储对话历史（实际生产中通常会存入 Redis 或数据库）
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 使用 RunnableWithMessageHistory 包装链条，自动处理历史记录的存取
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- 4. 运行测试 ---
config = {"configurable": {"session_id": "user_001"}}  # 指定一个用户会话ID

# 第一次对话
issue1 = "我买的鞋子尺码不合适，想换一双大一号的，怎么处理？"
result1 = memory_chain.invoke({"input": issue1}, config=config)
# 在链的外面，手动调用 parser 将文本转为 Pydantic 对象
result1 = parser.parse(result1.content)
print(f"第一次分析结果：{result1.model_dump_json(indent=2, ensure_ascii=False)}")

# 第二次对话（模型能结合上下文）
issue2 = "另外，换货的运费是谁承担呢？"
result2 = memory_chain.invoke({"input": issue2}, config=config)
# 在链的外面，手动调用 parser 将文本转为 Pydantic 对象
result2 = parser.parse(result2.content)
print(f"第二次分析结果：{result2.model_dump_json(indent=2, ensure_ascii=False)}")
