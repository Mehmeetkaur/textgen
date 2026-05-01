import json
import xml.etree.ElementTree as ET

from atria_types import AnnotatedObject, BoundingBox, Label
from atria_types._generic._doc_content import TextElement


def read_pascal_voc(
    xml_file: str, labels: list[str], image_width: int = None, image_height: int = None
) -> list[AnnotatedObject]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotated_object = []
    for object_ in root.iter("object"):
        ymin, xmin, ymax, xmax = None, None, None, None
        label = object_.find("name").text
        try:
            label = int(label)
        except:
            label = labels.index(label)
        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)
        annotated_object.append(
            AnnotatedObject(
                bbox=BoundingBox(value=[xmin, ymin, xmax, ymax]).ops.normalize(
                    width=image_width, height=image_height
                ),
                label=Label(value=label, name=labels[label]),
            )
        )
    return annotated_object


def read_words_json(
    words_file: str, image_width: int = None, image_height: int = None
) -> list[TextElement]:
    with open(words_file) as f:
        data = json.load(f)

    text_elements = []
    for word in data:
        text_elements.append(
            TextElement(
                text=word["text"],
                bbox=BoundingBox(value=word["bbox"]).ops.normalize(
                    width=image_width, height=image_height
                ),
            )
        )
    return text_elements
