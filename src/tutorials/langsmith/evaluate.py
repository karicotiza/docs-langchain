"""Tutorial module.

Evaluate a chatbot.
"""
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict, TypeVar

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama
from langsmith import Client
from langsmith.schemas import Dataset

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

client: Client = Client()

dataset_name: str = "QA Example Dataset"
# dataset: Dataset = client.create_dataset(dataset_name)

# client.create_examples(
#     inputs=[
#         {"question": "What is LangChain?"},
#         {"question": "What is LangSmith?"},
#         {"question": "What is OpenAI?"},
#         {"question": "What is Google?"},
#         {"question": "What is Mistral?"},
#     ],
#     outputs=[
#         {"answer": "A framework for building LLM applications"},
#         {"answer": "A platform for observing and evaluating LLM applications"},
#         {"answer": "A company that creates Large Language Models"},
#         {"answer": "A technology company known for search"},
#         {"answer": "A company that creates Large Language Models"},
#     ],
#     dataset_id=dataset.id,
# )

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)


class Role(StrEnum):
    """LangChain role adapter."""

    system = "system"
    human = "human"


@dataclass(slots=True)
class Message:
    """Message structure."""

    role: Role
    text: str


    def items(self) -> tuple[str, str]:  # noqa: WPS
        """Get structure as key, value.

        Returns:
            tuple[str, str]: role as a key and text as a value.

        """
        return (self.role, self.text)


def prompt(*args: Message) -> ChatPromptTemplate:
    """Make prompt for langchain.

    Args:
        args (list[Message]): list of tuple with roles and
            messages. You can set placeholders for values like "{value}", they
            will be invoked by model or chain later.

    Returns:
        ChatPromptTemplate: prepared ChatPromptTemplate.

    """
    return ChatPromptTemplate.from_messages(
        messages=[message.items() for message in args],
    )


GenericType = TypeVar("GenericType")


def _string(element: GenericType) -> str:
    if isinstance(element, str):
        return element

    msg: str = f"expected str, got {type(element)}"
    raise TypeError(msg)


class Inputs(TypedDict):
    """Inputs dict."""

    question: str


class ReferenceOutputs(TypedDict):
    """reference outputs."""

    answer: str


class Outputs(TypedDict):
    """Outputs dict."""

    response: str


correctness_prompt: ChatPromptTemplate = prompt(
    Message(
        role=Role.system,
        text=(
            "You are an expert professor specialized in grading students' "
            "answers to questions."
        ),
    ),
    Message(
        role=Role.human,
        text=(
            "You are grading the following question: {question}\n\n"
            "Here is the real answer: {answer}\n\n"
            "You are grading the following predicted answer: {response}\n\n"
            "Respond only with one word 'CORRECT' or 'INCORRECT'.\n"
            "Result: "
        ),
    ),
)

correctness_chain: RunnableSerializable[dict, BaseMessage] = (
    correctness_prompt
    | chat
)


def correctness(
    inputs: Inputs,
    reference_outputs: ReferenceOutputs,
    outputs: Outputs,
) -> bool:
    """Check answer correctness using LLM.

    Args:
        inputs (Inputs): Inputs TypedDict.
        reference_outputs (ReferenceOutputs): ReferenceOutputs TypedDict.
        outputs (Outputs): Outputs TypedDict.

    Returns:
        bool: True if reference_outputs and outputs is similar enough.

    """
    response: BaseMessage = correctness_chain.invoke(
        input={
            "question": inputs["question"],
            "answer": reference_outputs["answer"],
            "response": outputs["response"],
        },
    )

    return _string(response.content) == "CORRECT"


def concision(outputs: dict, reference_outputs: dict) -> bool:
    """Check answer concision.

    Args:
        outputs (dict): Outputs TypedDict.
        reference_outputs (dict): ReferenceOutputs TypedDict.

    Returns:
        bool: True if answer is concise.

    """
    expected_length: int = 2 * len(reference_outputs["answer"])
    actual_length: int = len(outputs["response"])

    return actual_length < expected_length


default_instructions: ChatPromptTemplate = prompt(
    Message(
        role=Role.system,
        text=(
            "Respond to the users question in a short, "
            "concise manner (one short sentence)."
        ),
    ),
    Message(
        role=Role.human,
        text=(
            "Question: {question}"
        ),
    ),
)

answering_chain: RunnableSerializable[dict, BaseMessage] = (
    default_instructions
    | chat
)


def my_app(question: str) -> str:
    """Get answer to question.

    Args:
        question (str): user's question.

    Returns:
        str: system's answer.

    """
    response: BaseMessage = answering_chain.invoke(
        input={
            "question": question,
        },
    )

    return _string(response.content)


def ls_target(inputs: Inputs) -> Outputs:
    """Target function for evaluate.

    Args:
        inputs (Inputs): Inputs TypedDict.

    Returns:
        Outputs: Outputs TypedDict.

    """
    return {
        "response": my_app(inputs["question"]),
    }
