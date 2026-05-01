IMPORT_CHECK = [
    # common types
    "ConfigType",
    "DatasetSplitType",
    "GANStage",
    "ModelType",
    "OCRType",
    "TaskType",
    "TrainingStage",
    # data instance types
    "BaseDataInstance",
    "DocumentInstance",
    "ImageInstance",
    # datasets metadata
    "DatasetLabels",
    "DatasetMetadata",
    "DatasetShardInfo",
    "SplitConfig",
    "SplitInfo",
    # generic types
    "AnnotatedObject",
    "BoundingBox",
    "BoundingBoxMode",
    "TextElement",
    "Annotation",
    "ClassificationAnnotation",
    "EntityLabelingAnnotation",
    "LayoutAnalysisAnnotation",
    "QuestionAnsweringAnnotation",
    "ObjectDetectionAnnotation",
    "Image",
    "Label",
    "OCR",
    "QAPair",
    "AnswerSpan",
]


def test_imports():
    for name in IMPORT_CHECK:
        module = __import__("atria_types", fromlist=[name])
        assert hasattr(module, name), f"Cannot import {name} from atria_types"
