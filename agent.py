from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel


class IntentAgent(BaseSingleActionAgent):
    tools: List
    llm: BaseLanguageModel
    intent_template: str = """
    现在有一些意图，类别为{intents}，你的任务是理解用户问题的意图，并判断该问题属于哪一类意图。
    回复的意图类别必须在提供的类别中，并且必须按格式回复：“意图类别：<>”。

    举例：
    问题：龙湖天下在哪？
    意图类别：新房信息问答
    
    问题：龙湖天下单价是多少？
    意图类别：新房信息问答
    
    问题：龙湖天下小区物业是？
    意图类别：新房信息问答
    
    问题：龙湖天下小区物业费多少？
    意图类别：新房信息问答
    
    问题：龙湖天下有哪些户型？
    意图类别：新房信息问答
    
    问题：预算300万，在南京买房有哪些推荐？
    意图类别：房地产复杂问题问答
    
    问题：同等预算，紫樾府和江悦润府怎么选？
    意图类别：房地产复杂问题问答
    
    问题：地铁1号线有哪些新房推荐？
    意图类别：房地产复杂问题问答

    问题：“{query}”
    """
    prompt = PromptTemplate.from_template(intent_template)
    llm_chain: LLMChain = None

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def choose_tools(self, query) -> List[str]:
        self.get_llm_chain()
        tool_names = [tool.name for tool in self.tools]
        resp = self.llm_chain.predict(intents=tool_names, query=query)
        select_tools = [(name, resp.index(name)) for name in tool_names if name in resp]
        select_tools.sort(key=lambda x: x[1])
        return [x[0] for x in select_tools]

    @property
    def input_keys(self):
        return ["input"]

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        # only for single tool
        tool_name = self.choose_tools(kwargs["input"])[0]
        return AgentAction(tool=tool_name, tool_input=kwargs["input"], log="")

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        raise NotImplementedError("IntentAgent does not support async")

