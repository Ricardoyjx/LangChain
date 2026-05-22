import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import ollama
import time

# 1.配置本地大模型端口
# 2.初始化大模型
llm = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434", temperature=0.7)

# 3.创建提示词模板
prompt = ChatPromptTemplate.from_template("请为以下主题写一首古诗:{topic}")

# 4.使用管道符 | 将提示词 -> 大模型 -> 输出解析器
chain = prompt | llm | StrOutputParser()


print("请输入主题:")
userInput = input()
# 5.调用链式调用
result = chain.invoke({"topic": userInput})
print(result)
