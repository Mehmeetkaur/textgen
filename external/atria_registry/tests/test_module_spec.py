import pytest
from omegaconf import ValidationError

from atria_registry.module_spec import DefaultKeyConflictError, ModuleSpec


def test_spec_mockclass2_valid():
    spec = ModuleSpec(
        name="test_module_1",
        module="utilities.MockClass2",
        group="test_group_1",
        var1=1,
        var2=2,
    )
    config = spec.build_config()
    assert isinstance(config, dict)


def test_spec_mockclass3_valid():
    spec = ModuleSpec(
        name="test_module_2",
        module="utilities.MockClass3",
        group="test_group_2",
        var1=1,
        var2=2,
    )
    config = spec.build_config()
    assert isinstance(config, dict)


def test_spec_mockclass4_valid_basic():
    spec = ModuleSpec(
        name="test_module_3",
        module="utilities.MockClass4",
        group="test_group_3",
        is_global_package=True,
        registers_target=True,
        var1=1,
        var2=2,
    )
    config = spec.build_config()
    assert isinstance(config, dict)


def test_spec_mockclass4_invalid_type_raises():
    spec = ModuleSpec(
        name="test_module_3",
        module="utilities.MockClass4",
        group="test_group_3",
        is_global_package=True,
        registers_target=True,
        var1="wrong_type",
        var2="wrong_type",
    )
    with pytest.raises(ValidationError):
        spec.build_config()


def test_spec_mockclass4_partial_valid():
    spec = ModuleSpec(
        name="test_module_3",
        module="utilities.MockClass4",
        group="test_group_3",
        is_global_package=True,
        registers_target=True,
        var1=1,
        var2=2,
    )
    config = spec.build_config(zen_partial=True)
    assert isinstance(config, dict)


def test_spec_mockclass4_with_extra_non_class_params():
    spec = ModuleSpec(
        name="test_module_3",
        module="utilities.MockClass4",
        group="test_group_3",
        is_global_package=True,
        registers_target=True,
        var1=1,
        var2=2,
        defaults=["_self_", {"/parent@child": "test"}],
        extra_param="extra_value",
    )
    with pytest.raises(TypeError):
        spec.build_config(zen_partial=True)


def test_spec_mockclass4_with_extra_allowed_on_module():
    spec = ModuleSpec(
        name="test_module_3",
        module="utilities.MockPydanticExtraAllowedClass",
        group="test_group_3",
        is_global_package=True,
        registers_target=True,
        var1=1,
        var2=2,
        defaults=["_self_", {"/parent@child": "test"}],
        extra_param="extra_value",
    )
    config = spec.build_config(zen_partial=True)
    assert isinstance(config, dict)
    assert config["extra_param"] == "extra_value"


def test_spec_mockclass4_with_defaults():
    spec = ModuleSpec(
        name="test_module_3",
        module="utilities.MockClass4",
        group="test_group_3",
        is_global_package=True,
        registers_target=True,
        var1=1,
        var2=2,
        defaults=["_self_", {"/parent@child": "test"}],
    )
    config = spec.build_config(zen_partial=True)
    assert "defaults" in config
    assert isinstance(config, dict)


def test_spec_mockclass4_with_conflicting_defaults():
    spec = ModuleSpec(
        name="test_module_3",
        module="utilities.MockClass4",
        group="test_group_3",
        is_global_package=True,
        registers_target=True,
        var1=1,
        var2=2,
        defaults=["_self_", {"/parent@child": "test"}, {"/parent@var1": "test"}],
    )
    with pytest.raises(DefaultKeyConflictError):
        config = spec.build_config(zen_partial=True)
        assert "defaults" in config
        assert isinstance(config, dict)
