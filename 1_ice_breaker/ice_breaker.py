import os
from typing import Tuple
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from output_parsers import Summary, summary_parser
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile


def ice_breaker_with(name:str) -> Tuple[Summary, str]:
    
    # Obtenemos la url a través de un crawler (Tavily), sabiendo el tipo de info que nos da
    # y lo pasamos por el llm para que filtre bien la url en caso de que haya varias, etc. 
    # Agente encargado de buscar la url correcta de linkedin
    linkedin_url = linkedin_lookup_agent(name=name)
    # Después, con al url, ya sí scrapeamos con la api de nubela
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url, mock=True)


    summary_template = """
        Given the Linkedin information {information} about a person, I want you to create:
        1. A short summary
        2. Two interesting facts about them
        \n{format_instructions}
    """

    # template para los prompts, que reciben una variable, con la información variable y
    # la template en sí. Es de lo más simple que veremos
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template,
        # Añadimos el parser de salida como partial_variables
        partial_variables={
            "format_instructions":summary_parser.get_format_instructions()
            }
    )
    # variable que contiene al llm. La temperatura indicará al LLM cuan creativo ha de ser.
    # Cuanta más temperatura más creativo será, pudiendo llegar a tener "hallucinations",
    # que es uno de los problemas más gordos de los LLMs, ya que es complicado hacerles dar
    # una respuesta EXACTA todo el tiempo, y es posible que se vayan por las ramas en sus
    # respuestas, eso es la alucinación.
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # Una cadena, cdnde concatenamos el summary y después el llm.
    # Con este pipie lo que hacemos es que coge el prompt que ha generado ese template con la info pasada
    # y hace una consulta al llm
    chain = (
        summary_prompt_template | llm | summary_parser
    )  # También se puede hacer con chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    
    # Ejecución de la consulta
    res: Summary = chain.invoke(input={"information": linkedin_data})
    return res, linkedin_data.get("profile_pic_url")




if __name__ == "__main__":

    load_dotenv()
    # print(os.getenv("OPENAI_API_KEY"))
    print("Ice Breaker Enter")
    res = ice_breaker_with(name="Domingo López Pacheco")
    print(res)


