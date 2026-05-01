from typing import Self

from atria_types._base._data_model import _load_any
from atria_types._data_instance._base import BaseDataInstance
from atria_types._data_instance._visualizers._base import Visualizer
from atria_types._data_instance._visualizers._document import DocumentVisualizer
from atria_types._generic._doc_content import DocumentContent
from atria_types._generic._image import Image
from atria_types._generic._ocr import OCR
from atria_types._generic._pdf import PDF
from atria_types._utilities._ocr_processors._base import OCRProcessor
from pydantic import model_validator


class DocumentInstance(BaseDataInstance):
    page_id: int | None = None
    pdf: PDF | None = None
    image: Image | None = None
    ocr: OCR | None = None
    content: DocumentContent | None = None

    @property
    def viz(self) -> Visualizer:
        return DocumentVisualizer(instance=self)

    @model_validator(mode="after")
    def validate_fields(self) -> "Self":
        # Ensure we have either image or PDF
        if self.image is None and self.pdf is None:
            raise ValueError("Either image or pdf must be provided")
        return self

        return self

    def load(
        self,
    ) -> Self:
        loaded_fields = {}
        for name in self.__class__.model_fields:
            loaded_fields[name] = _load_any(getattr(self, name))
        if (
            loaded_fields["ocr"] is not None
            and loaded_fields["ocr"].content is not None
            and loaded_fields["content"] is None
        ):
            loaded_fields["content"] = OCRProcessor.parse(
                raw_ocr=loaded_fields["ocr"].content, ocr_type=loaded_fields["ocr"].type
            )

        new_instance = self.model_copy(update=loaded_fields)
        return new_instance
