"""Test module.

Evaluate a chatbot.
"""


from typing import TYPE_CHECKING

from src.tutorials.langsmith.evaluate import (
    client,
    concision,
    correctness,
    dataset_name,
    ls_target,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langsmith.evaluation._runner import ExperimentResults
    from langsmith.schemas import SCORE_TYPE, Feedback


def test_evaluate() -> None:
    """Test evaluate."""
    experiment_prefix: str = "LLaMa3.2:3B"
    expected_score: float = 0.2

    experiment_results: ExperimentResults = client.evaluate(
        ls_target,
        data=dataset_name,
        evaluators=[
            concision,
            correctness,
        ],
        experiment_prefix=experiment_prefix,
    )

    feedbacks: Iterator[Feedback] = client.list_feedback(
        run_ids=[
            run.id
            for run
            in client.list_runs(
                project_name=experiment_results.experiment_name,
            )
        ],
        feedback_key="concision",
    )

    scores: list[SCORE_TYPE] = [feedback.score for feedback in feedbacks]

    assert sum(scores) / len(scores) >= expected_score
