# -*- coding: utf-8 -*-
"""
python 3.10


测试环境：
langchain==1.2.2
langchain-community==0.4.1
langchain-core==1.2.6
langsmith==0.6.1
"""
import os
from datetime import datetime

# from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import Tool
print('success')

from langchain_classic.memory import ConversationBufferMemory
# from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatZhipuAI

# 1. 配置 API Key
os.environ["ZHIPUAI_API_KEY"] = ""

# 2. 定义工具
def get_current_time(*args, **kwargs) -> str:
    return f"当前时间是 {datetime.now():%Y年%m月%d日 %H:%M:%S}"

tools = [
    Tool(
        name="GetTime",
        func=get_current_time,
        description="获取当前精确时间间。当用户问‘几点’/‘时间’等关键词时，必须立即调用本工具，不得反问。"
    )
]

# 3. 初始化 LLM
llm = ChatZhipuAI(model="glm-4-plus", temperature=0.01, verbose=True)

# 4. 构造 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有记忆的智能助手，能回答问题并调用工具。如果用户问时间，立即调用 GetTime 工具并直接返回工具结果，不要加任何解释或问候。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 5. 记忆组件
memory = ConversationBufferMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True
)

# 6. 构造 Agent & Executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# 7. 简单对话测试
if __name__ == "__main__":
    print("=== 第一轮 ===")
    print(agent_executor.invoke({"input": "你好！"}))
    print("\n=== 第二轮 ===")
    print(agent_executor.invoke({"input": "现在几点了？"}))
    print("\n=== 第三轮 ===")
    print(agent_executor.invoke({"input": "刚才的时间再告诉我一次？"}))

    # print("\n=== 第四轮 ===")
    # print(agent_executor.invoke({"input": "请调用工具获取时间"}))