"""Example: Creating and using Label objects."""

from atria_logger import get_logger

from atria_types import Label

logger = get_logger(__name__)


def main() -> None:
    # Create a simple label
    label = Label(value=1, name="person")
    logger.info("Simple Label:")
    logger.info(label)

    # Create labels for different classes
    labels = [
        Label(value=0, name="background"),
        Label(value=1, name="person"),
        Label(value=2, name="car"),
        Label(value=3, name="building"),
    ]

    logger.info("All labels:\n%s", labels)

    # Convert to dict
    label_dict = label.model_dump()
    logger.info("As dictionary:")
    logger.info(label_dict)

    # Serialize to JSON
    label_json = label.model_dump_json()
    logger.info("As JSON:")
    logger.info(label_json)


if __name__ == "__main__":
    main()
