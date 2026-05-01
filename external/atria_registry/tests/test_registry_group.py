from utilities import MockClass1

from atria_registry.module_spec import ModuleSpec
from atria_registry.registry_group import RegistryGroup


def test_file_store_build(test_registry_group: RegistryGroup):
    test_registry_group.register("mock_module", builds_as_yaml_file=True)(MockClass1)
    node = test_registry_group.store._open(
        f"{test_registry_group.name}/mock_module.yaml"
    )
    assert node is None

    test_registry_group.enable_file_store_build()
    test_registry_group.register("mock_module", builds_as_yaml_file=True)(MockClass1)
    node = test_registry_group.store._open(
        f"{test_registry_group.name}/mock_module.yaml"
    )
    assert node is not None


def test_basic_node_build(test_registry_group: RegistryGroup):
    test_registry_group.register("mock_module", builds_as_yaml_file=False)(MockClass1)
    node = test_registry_group.store._open(
        f"{test_registry_group.name}/mock_module.yaml"
    )
    assert node.group == "mock_group"
    assert node.name == "mock_module.yaml"
    assert node.node._target_ == "utilities.MockClass1"
    assert not node.node._partial_
    assert node.node.defaults == []
    assert node.package is None
    assert node.provider == test_registry_group._default_provider


def test_multiple_node_build(
    test_registry_group: RegistryGroup, test_specs: list[ModuleSpec]
):
    test_registry_group.register_modules(
        test_specs,
        group="mock_group",
        provider="test_provider",
        is_global_package=False,
    )
    for spec in test_specs:
        node = test_registry_group.store._open(
            f"{test_registry_group.name}/{spec.name}.yaml"
        )
        assert node.group == "mock_group"
        assert node.name == f"{spec.name}.yaml"
        assert node.node._target_ == spec.module
        assert not node.node._partial_
        assert node.node.defaults == []
        assert node.package is None
        assert node.provider == "test_provider"
