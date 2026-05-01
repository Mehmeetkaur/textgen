import bs4

from atria_types._generic._bounding_box import BoundingBox
from atria_types._generic._doc_content import DocumentContent, TextElement


class HOCRProcessor:
    @staticmethod
    def parse(raw_ocr: str) -> list[TextElement]:
        soup = bs4.BeautifulSoup(raw_ocr, features="xml")

        # Extract image size
        pages = soup.findAll("div", {"class": "ocr_page"})
        image_size_str = pages[0]["title"].split("; bbox")[1]
        w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())

        # Extract words and their properties
        text_elements: list[TextElement] = []
        ocr_words = soup.findAll("span", {"class": "ocrx_word"})
        for word in ocr_words:
            title = word["title"]
            conf = float(title[title.find(";") + 10 :])
            if word.text.strip() == "":
                continue

            # Get text angle from line title
            textangle = 0.0
            parent_title = word.parent["title"]
            if "textangle" in parent_title:
                textangle = float(parent_title.split("textangle")[1][1:3])

            x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
            text_elements.append(
                TextElement(
                    text=word.text.strip(),
                    bbox=BoundingBox(
                        value=[x1 / w, y1 / h, x2 / w, y2 / h], normalized=True
                    ),
                    conf=conf,
                    angle=textangle,
                )
            )

        return DocumentContent(
            text_elements=text_elements,
        )


    @staticmethod
    def parse_as_graph(raw_ocr: str) -> list[TextElement]:
        soup = bs4.BeautifulSoup(raw_ocr, features="xml")

        # Extract image size
        pages = soup.findAll("div", {"class": "ocr_page"})
        image_size_str = pages[0]["title"].split("; bbox")[1]
        w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())

        # Extract words and their properties
        text_elements: list[TextElement] = []
        ocr_words = soup.findAll("span", {"class": "ocrx_word"})
        for word in ocr_words:
            title = word["title"]
            conf = float(title[title.find(";") + 10 :])
            if word.text.strip() == "":
                continue

            # Get text angle from line title
            textangle = 0.0
            parent_title = word.parent["title"]
            if "textangle" in parent_title:
                textangle = float(parent_title.split("textangle")[1][1:3])

            x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
            text_elements.append(
                TextElement(
                    text=word.text.strip(),
                    bbox=BoundingBox(
                        value=[x1 / w, y1 / h, x2 / w, y2 / h], normalized=True
                    ),
                    conf=conf,
                    angle=textangle,
                )
            )

        return DocumentContent(
            text_elements=text_elements,
        )
