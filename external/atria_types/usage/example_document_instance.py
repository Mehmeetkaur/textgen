"""Example: Creating and using DocumentInstance (document understanding)."""

from atria_logger import get_logger
from PIL import Image as PILImage

from atria_types import PDF, DocumentContent, DocumentInstance, Image, TextElement
from atria_types._generic._bounding_box import BoundingBox

logger = get_logger(__name__)


def main() -> None:
    # Create a simple document instance from an image
    doc_from_image = DocumentInstance(
        sample_id="doc_001",
        page_id=0,
        image=Image(
            file_path="/data/docs/invoice_page1.jpg",
            content=PILImage.new("RGB", (2100, 2970)),  # A4 at 300 DPI
        ),
        content=DocumentContent(
            text_elements=[
                TextElement(
                    text="Invoice", bbox=BoundingBox(value=[100.0, 50.0, 300.0, 100.0])
                ),
                TextElement(
                    text="Date: 2024-01-15",
                    bbox=BoundingBox(value=[100.0, 150.0, 400.0, 180.0]),
                ),
                TextElement(
                    text="Amount: $1,250.00",
                    bbox=BoundingBox(value=[100.0, 200.0, 450.0, 230.0]),
                ),
            ],
        ),
    )

    logger.info("Example Document Instance:\n%s", doc_from_image)

    # Create a document instance from PDF
    doc_from_pdf = DocumentInstance(
        sample_id="doc_002",
        page_id=1,
        pdf=PDF(
            file_path="/data/docs/report.pdf",
            num_pages=1,
        ),
        image=Image(
            file_path="/data/docs/report_page1.png",
        ),
        content=DocumentContent(
            text="Annual Report 2024\n\nExecutive Summary\nThis report outlines...",
        ),
    )

    logger.info("Example PDF Document Instance:\n%s", doc_from_pdf)


if __name__ == "__main__":
    main()
