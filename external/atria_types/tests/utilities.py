from atria_logger import get_logger

logger = get_logger(__name__)


def _assert_values_equal(value1, value2, float_rtolerance=1e-05):
    """
    Compare two values.
        - For floats, uses math.isclose with the provided relative tolerance.
        - For lists, recursively compares each corresponding element.
        - For BaseModel instances, recurses into assert_models_equal.
        - For other types, compares with equality.
    """
    import math

    import numpy as np
    import torch
    from pydantic import BaseModel

    # Compare floats with tolerance.
    if isinstance(value1, float) and isinstance(value2, float):
        assert math.isclose(value1, value2, rel_tol=float_rtolerance), (
            f"Float values not close: {value1} vs {value2}"
        )
    # Recursively compare lists.
    elif isinstance(value1, list) and isinstance(value2, list):
        assert len(value1) == len(value2), (
            f"List lengths don't match: {len(value1)} vs {len(value2)}"
        )
        for item1, item2 in zip(value1, value2, strict=True):
            _assert_values_equal(item1, item2, float_rtolerance)
    elif isinstance(value1, dict) and isinstance(value2, dict):
        assert value1.keys() == value2.keys(), (
            f"Dict keys differ: {value1.keys()} vs {value2.keys()}"
        )
        for key in value1:
            _assert_values_equal(value1[key], value2[key], float_rtolerance)
    elif isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        assert value1.shape == value2.shape, (
            f"Tensor shapes differ: {value1.shape} vs {value2.shape}"
        )
        assert value1.dtype == value2.dtype, (
            f"Tensor dtypes differ: {value1.dtype} vs {value2.dtype}"
        )
        assert torch.allclose(value1, value2, rtol=float_rtolerance), (
            f"Tensors not close: {value1} vs {value2}"
        )
    elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        assert value1.shape == value2.shape, (
            f"Tensor shapes differ: {value1.shape} vs {value2.shape}"
        )
        assert value1.dtype == value2.dtype, (
            f"Tensor dtypes differ: {value1.dtype} vs {value2.dtype}"
        )
        assert np.allclose(value1, value2, rtol=float_rtolerance), (
            f"Tensors not close: {value1} vs {value2}"
        )
    # Recurse into nested pydantic models.
    elif isinstance(value1, BaseModel) and isinstance(value2, BaseModel):
        _assert_models_equal(value1, value2, float_rtolerance)
    else:
        assert value1 == value2, f"Values do not match: {value1} vs {value2}"


def _assert_attribute_types_equal(model1, model2):
    """
    Checks that the common attributes in both models have the same type.
    Iterates over the model fields as defined by .__fields__.
    """

    from pydantic import BaseModel

    for field in model1.model_fields_set:
        # If the field exists in both models.
        if hasattr(model2, field):
            type1 = type(getattr(model1, field))
            type2 = type(getattr(model2, field))
            if not issubclass(type1, BaseModel):
                assert type1 == type2, (
                    f"Type mismatch for field '{field}': {type1} vs {type2}"
                )


def _assert_models_values_equal(model1, model2, float_rtolerance=1e-05):
    """
    Checks that two models have the same values.
    It uses compare_values to handle floats (with tolerance), lists, and nested models.
    """
    # Compare using the dict representations. This assumes that both models return the same keys.
    for key in model1.model_fields_set:
        _assert_values_equal(
            getattr(model1, key), getattr(model2, key), float_rtolerance
        )


def _assert_models_equal(model1, model2, float_rtolerance=1e-05):
    """
    Checks that two pydantic models have:
        1. Attributes of the same type.
        2. Close enough values.
    """
    _assert_attribute_types_equal(model1, model2)
    _assert_models_values_equal(model1, model2, float_rtolerance)


def _compare_dicts_recursively(dict1, dict2, path=""):
    differences = []
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else str(key)

        if key not in dict1:
            differences.append(f"Missing key in actual: {current_path}")
        elif key not in dict2:
            differences.append(f"Unexpected key in actual: {current_path}")
        elif dict1[key] != dict2[key]:
            differences.append(
                f"Value mismatch at {current_path}: expected {dict2[key]}, got {dict1[key]}"
            )

    return differences
