from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from functools import partial
from typing import List, Optional, Mapping, Any
import openai


# 基于vllm推理
class LL:
    def __init__(self, model, api_base=None, api_key=None):
        self.model = model
        self.api_base = 'http://192.168.204.120:7000/v1'
        self.api_key = 'EMPTY'

    def stream_chat(
            self,
            prompt,
            history=[],
            max_length=3096,
            top_p=0.7,
            temperature=0.1
    ):
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            stream=False,
            stop=["</s>", '<|im_end|>', '<|endoftext|>'],
            temperature=temperature,
            use_beam_search=False, n=1, frequency_penalty=1.0, best_of=1,
            presence_penalty=1.0, top_p=top_p, top_k=1, max_tokens=max_length
        )
        response = completion['choices'][0]['text']

        history.append([prompt, response])
        yield response, history

    def chat(self, prompt,
             history=[],
             max_length=3096,
             top_p=0.7,
             temperature=0.1
             ):
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            stream=False,
            stop=["</s>", '<|im_end|>', '<|endoftext|>'],
            temperature=temperature,
            use_beam_search=False, n=1, frequency_penalty=1.0, best_of=1,
            presence_penalty=1.0, top_p=top_p, top_k=1, max_tokens=max_length
        )
        respose = completion['choices'][0]['text']

        history.append([prompt, respose])

        return respose, history


class LLMs(LLM):
    model_path: str
    max_length: int = 2048
    temperature: float = 0.1
    top_p: float = 0.7
    history: List = []
    streaming: bool = True
    model: object = None

    @property
    def _llm_type(self) -> str:
        return "llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history": [],
            "streaming": self.streaming
        }

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            add_history: bool = False
    ) -> str:
        if self.model is None:
            raise RuntimeError("Must call `load_model()` to load model and tokenizer!")

        if self.streaming:
            text_callback = partial(StreamingStdOutCallbackHandler().on_llm_new_token, verbose=True)
            resp = self.generate_resp(prompt, text_callback, add_history=add_history)
        else:
            resp = self.generate_resp(self, prompt, add_history=add_history)

        return resp

    def generate_resp(self, prompt, text_callback=None, add_history=True):
        resp = ""
        index = 0
        if text_callback:
            for i, (resp, _) in enumerate(self.model.stream_chat(
                    prompt,
                    self.history,
                    max_length=self.max_length,
                    top_p=self.top_p,
                    temperature=self.temperature
            )):
                if add_history:
                    if i == 0:
                        self.history += [[prompt, resp]]
                    else:
                        self.history[-1] = [prompt, resp]
                text_callback(resp[index:])
                index = len(resp)
        else:
            resp, _ = self.model.chat(
                prompt,
                self.history,
                max_length=self.max_length,
                top_p=self.top_p,
                temperature=self.temperature
            )
            if add_history:
                self.history += [[prompt, resp]]
        return resp

    def load_model(self):
        if self.model is not None:
            return
        self.model = LL(self.model_path)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._identifying_params:
                self.k = v
