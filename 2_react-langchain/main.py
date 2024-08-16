from typing import List, Union
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.tools import Tool
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str

from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)

from callbacks import AgentCallbackHandler


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    # Cuando ponemos {text=} generará un output de text=<variable_text>
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')  # Quitando saltos de línea y comillas

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    # El agent_scratchpad tiene toda la historia y la información de ejecución reactiva anterior

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # Añadimos token de parada, que es "Observation". Cuando produzca la Observación
    # para de generar resultados el llm
    llm = ChatOpenAI(
        temperature=0, stop=["Observation"], callbacks=[AgentCallbackHandler()]
    )

    intermediate_steps = []

    # Crea el input del prompt inicial, luego se lo pasa al prompt. Luego el prompt al LLM
    # generando hasta el stop word Observation, luego lo parseo,
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        # Ejecutamos la cadena y el resultado lo analizo
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the text in characters of the text DOG?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(agent_step)
        # Si es un AgentAction, tendrá la herramienta que va a utilizar. Compruebo que la herramiena existe y no se la ha inventado
        # Si existe, hago uso de tool_to_use.func(str(tool_input)) -> Ejecútame esta función "tool_to_use" con el input tool_input
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
