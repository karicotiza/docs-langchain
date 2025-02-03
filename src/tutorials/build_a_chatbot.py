"""Tutorial module.

Build a chatbot.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

if TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue

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

polyglot_system_message_prompt_template: SystemMessagePromptTemplate = (
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability in {language}.",
    )
)

polyglot_prompt_template: ChatPromptTemplate = (
    ChatPromptTemplate.from_messages(
        [
            polyglot_system_message_prompt_template,
            MessagesPlaceholder(variable_name="messages"),
        ],
    )
)


class PolyglotState(TypedDict):
    """Polyglot workflow state."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


polyglot_workflow: StateGraph = StateGraph(state_schema=PolyglotState)


def polyglot_call_model(state: PolyglotState) -> dict[str, BaseMessage]:
    """Call polyglot model.

    Args:
        state (PolyglotState): polyglot workflow state.

    Returns:
        dict[str, BaseMessage]: model response.

    """
    prompt: PromptValue = polyglot_prompt_template.invoke(state)
    response: BaseMessage = chat.invoke(prompt)
    return {"messages": response}


polyglot_workflow.add_edge(START, "model")
polyglot_workflow.add_node("model", polyglot_call_model)

polyglot_memory: MemorySaver = MemorySaver()
polyglot_app: CompiledStateGraph = polyglot_workflow.compile(
    checkpointer=polyglot_memory,
)

max_tokens: int = 60

trimmer: Any = trim_messages(
    max_tokens=max_tokens,
    strategy="last",
    token_counter=chat,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages_to_trim: list[BaseMessage] = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]


trimmed_workflow: StateGraph = StateGraph(state_schema=PolyglotState)


def trimmed_call_model(state: PolyglotState) -> dict[str, BaseMessage]:
    """Call trimmed model.

    Args:
        state (PolyglotState): trimmed workflow state.

    Returns:
        dict[str, BaseMessage]: model response.

    """
    trimmed_messages: Any = trimmer.invoke(state["messages"])
    prompt: PromptValue = polyglot_prompt_template.invoke(
        {
            "messages": trimmed_messages,
            "language": state["language"],
        },
    )

    response: BaseMessage = chat.invoke(prompt)
    return {"messages": response}


trimmed_workflow.add_edge(START, "model")
trimmed_workflow.add_node("model", trimmed_call_model)

trimmed_memory: MemorySaver = MemorySaver()
trimmed_app: CompiledStateGraph = trimmed_workflow.compile(
    checkpointer=trimmed_memory,
)
