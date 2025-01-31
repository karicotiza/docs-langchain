"""Tutorial module.

Build a chatbot.
"""

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import PromptValue
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


pirate_system_prompt: SystemMessage = SystemMessage(
    "You talk like a pirate. "
    "Answer all questions to the best of your ability.",
)

prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        pirate_system_prompt,
        MessagesPlaceholder(variable_name="messages"),
    ],
)

pirate_workflow: StateGraph = StateGraph(state_schema=MessagesState)


def pirate_call_model(state: MessagesState) -> dict[str, BaseMessage]:
    """Call pirate model.

    Args:
        state (MessagesState): workflow state.

    Returns:
        dict[str, BaseMessage]: model response.

    """
    prompt: PromptValue = prompt_template.invoke(state)
    response: BaseMessage = chat.invoke(prompt)
    return {"messages": response}


pirate_workflow.add_edge(START, "model")
pirate_workflow.add_node("model", pirate_call_model)

pirate_memory: MemorySaver = MemorySaver()
pirate_app: CompiledStateGraph = pirate_workflow.compile(
    checkpointer=pirate_memory,
)
