from typing import Annotated

from pydantic import field_serializer, field_validator

from atria_types._base._data_model import BaseDataModel
from atria_types._pydantic import (
    IntField,
    StrField,
    TableSchemaMetadata,
)


class AnswerSpan(BaseDataModel):
    start: IntField
    end: IntField
    text: StrField


class QAPair(BaseDataModel):
    id: IntField
    question_text: StrField
    answer_spans: Annotated[
        list[AnswerSpan] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @property
    def answers(self) -> list[str]:
        if not self.answer_spans:
            return []
        return [span.text for span in self.answer_spans]

    @field_validator("answer_spans", mode="before")
    def validate_answer_spans(cls, value) -> list[AnswerSpan]:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("answer_spans")
    def serialize_answer_spans(self, value: list[AnswerSpan]) -> str:
        import json

        return json.dumps([item.model_dump() for item in value])
