"""Tutorial module.

Build an Agent.
"""

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledGraph
from langgraph.prebuilt import create_react_agent

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

max_result: int = 50
search: DuckDuckGoSearchResults = DuckDuckGoSearchResults()
tools: list[BaseTool] = [search]

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

memory: MemorySaver = MemorySaver()

stateless_agent_executor: CompiledGraph = create_react_agent(
    model=chat,
    tools=tools,
)

stateful_agent_executor: CompiledGraph = create_react_agent(
    model=chat,
    tools=tools,
    checkpointer=memory,
)
