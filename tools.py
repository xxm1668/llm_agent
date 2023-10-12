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
    name = "游戏角色信息查询"
    description = "存有一些角色和信息的工具，输入应该是对游戏角色的询问"

    # QA params
    context = "已知游戏角色信息：  Mario: 马里奥是日本电子游戏设计师宫本茂创作的一个角色。他是同名电子游戏系列的主角，也是日本电子游戏公司任天堂的吉祥物。Princess Peach: 碧姬公主，是任天堂著名游戏系列马里奥系列中的重要角色。她是游戏中虚构的蘑菇王国的公主，也是王国的统治者。"
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
        context = "已知游戏角色信息：  Mario: 马里奥是日本电子游戏设计师宫本茂创作的一个角色。他是同名电子游戏系列的主角，也是日本电子游戏公司任天堂的吉祥物。Princess Peach: 碧姬公主，是任天堂著名游戏系列马里奥系列中的重要角色。她是游戏中虚构的蘑菇王国的公主，也是王国的统治者。"
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)


class Actor_knowledge_Tool(functional_Tool):
    llm: BaseLanguageModel

    # tool description
    name = "演员信息查询"
    description = "存有一些演员的工具，输入应该是对演员的询问"

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
        context = "已知演员信息：  梁朝伟: 1962年6月27日出生于中国香港，祖籍广东台山，华语影视男演员、歌手，国家一级演员, 汤唯: 1979年10月7日出生于浙江省杭州市，毕业于中央戏剧学院导演系本科班，中国内地女演员。"
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
