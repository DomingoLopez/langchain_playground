from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


# Clase sobre la que podemos hacer override de sus métodos heredando de la interface BaseCallbackHandler
# para hacerlos nuestros.
class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

        # Aquí añadimos lo que queremos que haga al iniciar la llamada al LLM
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("********")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("********")
