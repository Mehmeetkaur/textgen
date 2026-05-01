import pytest

from atria_registry import RegistryGroup


@pytest.fixture(scope="class", autouse=True)
def test_registry_group():
    return RegistryGroup(name="mock_group", default_provider="dataset")


@pytest.fixture(scope="class")
def test_specs():
    from atria_registry.module_spec import ModuleSpec

    return [
        ModuleSpec(module="utilities.MockClass1", name="mock_class_1"),
        ModuleSpec(module="utilities.MockClass2", name="mock_class_2"),
    ]
