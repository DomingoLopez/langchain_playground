import os, sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from tools.tools import get_profile_url_tavily
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Para ejecutar el modulito sería python -m agents.linkedin_lookup_agent 

# Carga de variables de entorno
load_dotenv()


def lookup(name:str) -> str:
    """
    Función para buscar un {name} en linkedin y devolver su url

    Args:
        name (str): nombre de la persona

    Returns:
        str: url del perfil de linkedin
    """
    
    # Declaramos el modelo a usar
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )
    
    # Declaramos el template
    template = """Given the full name {name_of_person}, I want you to get a link to their Linkeding Profile page. Your answer should contain only a URL."""
    
    # Declaramos objeto template con las variables de entrada
    prompt_template = PromptTemplate(
        template=template, 
        input_variables=["name_of_person"]
    )
    
    # Tools para el agente que podrá usar
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin page",
            func=get_profile_url_tavily,
            description="useful for when you need the Linkedin Page URL"
        )
    ]
    
    # Prompt inicial
    react_prompt = hub.pull("hwchase17/react")
    
    # Se crea el agente, con sus herramientas y el prompt inicial
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    # Se crea un ejecutor, que recibe un agente, las herramientas y verbose = True para ver alguna salida
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    # Se ejecuta el agente
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person = name)}
    )
    # Obtenemos el output
    linked_profile_url = result["output"]
    return linked_profile_url
    



if __name__ == "__main__":
    linkedin_url = lookup(name="Domingo López Pacheco")
    print(linkedin_url)