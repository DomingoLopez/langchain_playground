from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool

# Ojo con los paquetes de langchain_experimental. Cuidao con hacerlo en producción.
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent


load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # tool de intérprete de python
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    tools = [repl_tool]

    agent = create_react_agent(
        prompt=prompt, llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"), tools=tools
    )

    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # agent_executor.invoke(
    #     {
    #         "input": """
    #         generate and save in current working directory (inside new folder named qr_codes) 15 QRCodes that point to www.udemy.com/course/langchain,
    #         you have qrcode package installed already
    #         """
    #     }
    # )

    # -------------------------------------------------------------------------------

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    csv_agent.invoke({"input": "which season has the most episodes in file episode_info.csv"})
    # csv_agent.invoke({"input": "¿Cuántas episódios por año y por director se han rodado? usando el fichero episode_info.csv"})
    
    # Es muy posible que no mande el csv completo, ya que el context es más pequeño que el csv completo. 


if __name__ == "__main__":
    main()
