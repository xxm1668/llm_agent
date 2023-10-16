from langchain.agents import AgentExecutor
from tools import Character_knowledge_Tool, Base_knowledge_Tool
from agent import IntentAgent
from parse_out import CustomOutputParser
from langchain.memory import ConversationBufferMemory
from model2 import LLMs

output_parser = CustomOutputParser()
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='启动参数配置')
parser.add_argument('--model', type=str, help='model文件路径')
parser.add_argument('--max_length', type=int, help='期望模型处理的最大文本长度')
parser.add_argument('--temperature', type=str, help='调节模型输出多样性，越小输出越稳定')

# 解析命令行参数
args = parser.parse_args()
llm = LLMs(model_path=args.model, max_length=args.max_length, temperature=args.temperature)
llm.load_model()
tools = [Base_knowledge_Tool(llm=llm, return_direct=True), Character_knowledge_Tool(llm=llm, return_direct=True)]
agent = IntentAgent(tools=tools, llm=llm)
# 意图类别
# result = agent.choose_tools("江悦润府有哪些户型？")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1, memory=memory)
print('query: ', "预算400万，雨花台有哪些新房可以推荐？")
response = agent_exec.run("预算400万，雨花台有哪些新房可以推荐？")
print(response)
