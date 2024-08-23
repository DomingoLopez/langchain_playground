from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

@tool
def multiply(x:float, y:float)->float:
    """Multiply 'x' times 'y'."""
    return x*y




if __name__ == "__main__":
    print("Hello Tool Calling")
    

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful assitant"),
            ("human", "{input}"),
            ("placeholder","{agent_scratchpad}")
        ]
    )
    
    tools = [TavilySearchResults(), multiply]
    llm = ChatOpenAI(model="gpt-4-turbo")
    #llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_exectuor: AgentExecutor = AgentExecutor(agent=agent, tools=tools)
    
    res = agent_exectuor.invoke({
        "input":"what is the weather in dubai right now? Compare it with San Francisco, output should be in celsious."
    })
    
    print(res)