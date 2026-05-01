from atria_types._common import OCRType
from atria_types._generic._doc_content import DocumentContent
from atria_types._utilities._ocr_processors._hocr_processor import HOCRProcessor


class OCRProcessor:
    @staticmethod
    def parse(raw_ocr: str, ocr_type: OCRType) -> DocumentContent:
        if ocr_type == OCRType.tesseract:
            return HOCRProcessor.parse(raw_ocr)
        else:
            raise ValueError(f"Unsupported OCR type: {ocr_type}")
