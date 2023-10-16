from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class functional_Tool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""
    return_direct: bool = False

    def _call_func(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_func(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")


class Character_knowledge_Tool(functional_Tool):
    llm: BaseLanguageModel

    # tool description
    name = "房地产复杂问题问答"
    description = "输入应该是对预算情况下新房选择问答"
    return_direct = False

    # QA params
    qa_template = """
    我希望你扮演一个乐于助人、有礼貌及诚实的房地产置业顾问。
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_func(self, query) -> str:
        self.get_llm_chain()
        resp = self.llm_chain.predict(query=query)
        return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)


class Base_knowledge_Tool(functional_Tool):
    llm: BaseLanguageModel

    # tool description
    name = "新房信息问答"
    description = "输入应该是对新房地点、价格、单价、户型、建筑面积、商场、占地面积、小区绿化、小区物业、小区物业费的问答"
    return_direct = False
    # QA params
    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    如果该文本中没有相关内容可以回答问题，请直接回复：“抱歉，该问题需要更多上下文信息。”
    ```{text}```
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = "已知新房信息：  江悦润府位于主城北国家级经济开发区内，由改善大师正荣地产与京师国匠石榴集团合作开发，也是正荣继润江城、润峯等之后的又一“润”字力作。 江悦润府配套完善，生活氛围浓厚，交通便捷, 三纵三横主干道速达全城,地铁6、7号线在建，最快预计2021年通车。北侧规划66班制小学初中，周边还有栖霞区实验小学，前唐路小学、燕子矶中学等多所学府环伺。3km范围内坐拥金地汇峯中心华润万象天地和招商花园城等商业。江悦润府匠造12栋新智产品，主力76、89、105㎡阳光户型，精装交付，自带超5000㎡商业街区，赋新主城人居想象。"
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
