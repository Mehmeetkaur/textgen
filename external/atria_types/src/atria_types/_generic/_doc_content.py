from enum import Enum
from typing import Annotated, Any

from atria_types._base._data_model import BaseDataModel
from atria_types._generic._bounding_box import BoundingBox
from atria_types._pydantic import OptFloatField, OptStrField, TableSchemaMetadata
from pydantic import BaseModel, model_validator


class OCRLevel(str, Enum):
    PAGE = "page"
    BLOCK = "block"
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"


class OCRGraphNode(BaseModel):
    id: int | str
    word: str | None = None
    level: OCRLevel | None = None
    bbox: BoundingBox | None = None
    segment_level_bbox: BoundingBox | None = None
    conf: float | None = None
    angle: float | None = None


class OCRGraphLink(BaseModel):
    source: int | str
    target: int | str
    relation: str | None = None


class OCRGraph(BaseModel):
    directed: bool | None
    multigraph: bool | None
    graph: dict | None
    nodes: list[OCRGraphNode]
    links: list[OCRGraphLink]

    @property
    def words(self) -> list[str] | None:
        words = [node.word for node in self.nodes if node.level == OCRLevel.WORD]
        if any(word is None for word in words):
            return None
        return words  # type: ignore

    @property
    def word_bboxes(self) -> list[BoundingBox] | None:
        bboxes = [node.bbox for node in self.nodes if node.level == OCRLevel.WORD]
        if any(bbox is None for bbox in bboxes):
            return None
        return bboxes  # type: ignore

    @property
    def word_segment_level_bboxes(self) -> list[BoundingBox] | None:
        bboxes = [
            node.segment_level_bbox
            for node in self.nodes
            if node.level == OCRLevel.WORD
        ]
        if any(bbox is None for bbox in bboxes):
            return None
        return bboxes  # type: ignore

    @property
    def word_confs(self) -> list[float] | None:
        confs = [node.conf for node in self.nodes if node.level == OCRLevel.WORD]
        if any(conf is None for conf in confs):
            return None
        return confs  # type: ignore

    @property
    def word_angles(self) -> list[float] | None:
        angles = [node.angle for node in self.nodes if node.level == OCRLevel.WORD]
        if any(angle is None for angle in angles):
            return None
        return angles  # type: ignore


class TextElement(BaseDataModel):
    text: OptStrField = None
    bbox: BoundingBox | None = None
    segment_bbox: BoundingBox | None = None
    conf: OptFloatField = None
    angle: OptFloatField = None


class DocumentContent(BaseDataModel):
    text: OptStrField = None
    text_elements: Annotated[
        list[TextElement] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @property
    def text_list(self) -> list[str]:
        if self.text_elements is None:
            return []
        return [te.text for te in self.text_elements if te.text is not None]

    @property
    def bbox_list(self) -> list[BoundingBox]:
        if self.text_elements is None:
            return []
        return [te.bbox for te in self.text_elements if te.bbox is not None]

    @property
    def segment_bbox_list(self) -> list[BoundingBox]:
        if self.text_elements is None:
            return []
        return [
            te.segment_bbox for te in self.text_elements if te.segment_bbox is not None
        ]

    @model_validator(mode="before")
    def validate_content(cls, values: Any) -> Any:
        if values.get("text") is None and values.get("text_elements") is not None:
            texts = [te.text for te in values["text_elements"] if te.text is not None]
            values["text"] = " ".join(texts)
        return values
