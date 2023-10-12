from langchain.agents import AgentExecutor
from tools import Character_knowledge_Tool, Actor_knowledge_Tool
from model import LLMs
from agent import IntentAgent
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='启动参数配置')
parser.add_argument('--model', type=str, help='model文件路径')
parser.add_argument('--max_length', type=int, help='期望模型处理的最大文本长度')
parser.add_argument('--temperature', type=str, help='调节模型输出多样性，越小输出越稳定')

# 解析命令行参数
args = parser.parse_args()
llm = LLMs(model_path=args.model, max_length=args.max_length, temperature=args.temperature)
llm.load_model()
tools = [Character_knowledge_Tool(llm=llm), Actor_knowledge_Tool(llm=llm)]

agent = IntentAgent(tools=tools, llm=llm)
# result = agent.choose_tools("游戏角色马里奥是谁？")
agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
print('query: ', "游戏角色马里奥是谁？")
response = agent_exec.run("游戏角色马里奥是谁？")
