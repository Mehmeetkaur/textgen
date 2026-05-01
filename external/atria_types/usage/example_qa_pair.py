"""Example: Creating and using QAPair (Question-Answer pairs)."""

from atria_logger import get_logger

from atria_types import AnswerSpan, QAPair

logger = get_logger(__name__)


def main() -> None:
    # Context text for the QA
    context = "The Eiffel Tower is located in Paris, France. It was completed in 1889."

    # Create a simple QA pair with answer spans
    qa = QAPair(
        id=1,
        question_text="Where is the Eiffel Tower located?",
        answer_spans=[
            AnswerSpan(start=31, end=36, text="Paris"),
            AnswerSpan(start=38, end=44, text="France"),
        ],
    )

    logger.info("Simple QA Pair:")
    logger.info(qa)
    logger.info("Answer spans in context:")
    for span in qa.answer_spans:
        logger.info("  - '%s' at position %s:%s", span.text, span.start, span.end)
        logger.info("    Extracted: '%s'", context[span.start : span.end])

    # Create multiple QA pairs
    qa_pairs = [
        QAPair(
            id=1,
            question_text="Where is the Eiffel Tower?",
            answer_spans=[AnswerSpan(start=32, end=45, text="Paris, France")],
        ),
        QAPair(
            id=2,
            question_text="When was the Eiffel Tower completed?",
            answer_spans=[AnswerSpan(start=66, end=70, text="1889")],
        ),
    ]

    logger.info("All QA pairs:\n%s", qa_pairs)

    # Convert to dict
    qa_dict = qa.model_dump()
    logger.info("As dictionary:")
    logger.info(qa_dict)

    # Serialize to JSON
    qa_json = qa.model_dump_json()
    logger.info("As JSON:")
    logger.info(qa_json)


if __name__ == "__main__":
    main()
