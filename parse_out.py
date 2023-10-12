from typing import Dict, Union, Any, List

from langchain.output_parsers.json import parse_json_markdown
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.agents import AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish


# 自定义解析类
class CustomOutputParser(AgentOutputParser):

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print(text)
        cleaned_output = text.strip()
        # 定义匹配正则
        action_pattern = r'"action":\s*"([^"]*)"'
        action_input_pattern = r'"action_input":\s*"([^"]*)"'
        # 提取出匹配到的action值
        action = re.search(action_pattern, cleaned_output)
        action_input = re.search(action_input_pattern, cleaned_output)
        if action:
            action_value = action.group(1)
        if action_input:
            action_input_value = action_input.group(1)

        # 如果遇到'Final Answer'，则判断为本次提问的最终答案了
        if action_value and action_input_value:
            if action_value == "Final Answer":
                return AgentFinish({"output": action_input_value}, text)
            else:
                return AgentAction(action_value, action_input_value, text)

        # 如果声明的正则未匹配到，则用json格式进行匹配
        response = parse_json_markdown(text)

        action_value = response["action"]
        action_input_value = response["action_input"]
        if action_value == "Final Answer":
            return AgentFinish({"output": action_input_value}, text)
        else:
            return AgentAction(action_value, action_input_value, text)


output_parser = CustomOutputParser()
