from __future__ import annotations

import enum
from typing import Annotated, Literal

from pydantic import Field, field_serializer, field_validator

from atria_types._base._data_model import BaseDataModel
from atria_types._generic._annotated_object import AnnotatedObject
from atria_types._generic._label import Label
from atria_types._generic._qa_pair import QAPair
from atria_types._pydantic import TableSchemaMetadata


class AnnotationType(str, enum.Enum):
    classification = "classification"
    entity_labeling = "entity_labeling"
    question_answering = "question_answering"
    object_detection = "object_detection"
    layout_analysis = "layout_analysis"


class ClassificationAnnotation(BaseDataModel):
    type: Literal["classification"] = AnnotationType.classification.value
    label: Label


class EntityLabelingAnnotation(BaseDataModel):
    type: Literal["entity_labeling"] = AnnotationType.entity_labeling.value
    word_labels: list[Label]

    @field_validator("word_labels", mode="before")
    def validate_word_labels(cls, value) -> list[Label] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("word_labels")
    def serialize_word_labels(self, value: list[Label]) -> str | None:
        import json

        if value is not None:
            return json.dumps([item.model_dump() for item in value])
        return None


class QuestionAnsweringAnnotation(BaseDataModel):
    type: Literal["question_answering"] = AnnotationType.question_answering.value
    qa_pairs: Annotated[list[QAPair] | None, TableSchemaMetadata(pa_type="string")] = (
        None
    )

    @field_validator("qa_pairs", mode="before")
    def validate_qa_pairs(cls, value) -> list[QAPair] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("qa_pairs")
    def serialize_qa_pairs(self, value: list[QAPair]) -> str | None:
        import json

        if value is not None:
            return json.dumps([item.model_dump() for item in value])
        return None


class ObjectDetectionAnnotation(BaseDataModel):
    type: Literal["object_detection"] = AnnotationType.object_detection.value
    annotated_objects: Annotated[
        list[AnnotatedObject] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @field_validator("annotated_objects", mode="before")
    def validate_annotated_objects(cls, value) -> list[AnnotatedObject] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("annotated_objects")
    def serialize_annotated_objects(self, value: list[AnnotatedObject]) -> str | None:
        import json

        if value is not None:
            return json.dumps([item.model_dump() for item in value])
        return None


class LayoutAnalysisAnnotation(ObjectDetectionAnnotation):
    type: Literal["layout_analysis"] = AnnotationType.layout_analysis.value


Annotation = Annotated[
    ClassificationAnnotation
    | EntityLabelingAnnotation
    | QuestionAnsweringAnnotation
    | ObjectDetectionAnnotation
    | LayoutAnalysisAnnotation,
    Field(discriminator="type"),
]
