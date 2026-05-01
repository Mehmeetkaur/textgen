from atria_registry.registry_group_new import RegistryGroup


class MockClass:
    def __init__(self, var1: int, var2: int):
        self.var1 = var1
        self.var2 = var2


group = RegistryGroup(name="datasets", default_provider="atria")

print(group)
group.register(name="mock_class")(MockClass)
print(group)
