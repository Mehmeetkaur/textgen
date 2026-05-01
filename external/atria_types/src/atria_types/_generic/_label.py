from atria_types._base._data_model import BaseDataModel
from atria_types._pydantic import (
    IntField,
    StrField,
)


class Label(BaseDataModel):
    value: IntField
    name: StrField
