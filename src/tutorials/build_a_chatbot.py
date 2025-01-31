"""Tutorial module.

Build a chatbot.
"""

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

workflow: StateGraph = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState) -> dict[str, BaseMessage]:
    """Call model.

    Args:
        state (MessagesState): workflow state.

    Returns:
        dict[str, BaseMessage]: model response.

    """
    response: BaseMessage = chat.invoke(state["messages"])
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory: MemorySaver = MemorySaver()
app: CompiledStateGraph = workflow.compile(checkpointer=memory)
