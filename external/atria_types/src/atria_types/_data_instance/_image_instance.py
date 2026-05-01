from atria_types._data_instance._base import BaseDataInstance
from atria_types._generic._image import Image


class ImageInstance(BaseDataInstance):  # type: ignore[misc]
    image: Image
