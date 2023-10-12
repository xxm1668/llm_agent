from langchain.agents import AgentExecutor
from tools import Character_knowledge_Tool, Actor_knowledge_Tool
from model import ChatGLM
from agent import IntentAgent

llm = ChatGLM(model_path="/home/xxm/model/new/chatglm-6b")
llm.load_model()
tools = [Character_knowledge_Tool(llm=llm), Actor_knowledge_Tool(llm=llm)]

agent = IntentAgent(tools=tools, llm=llm)
# agent.choose_tools("游戏角色马里奥是谁？")
agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
agent_exec.run("游戏角色马里奥是谁？")
