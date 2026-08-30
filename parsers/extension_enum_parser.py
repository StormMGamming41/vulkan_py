from model import Registry, Enum_Value, C_Type
from .base import Base_Parser

BASE_VALUE = 1_000_000_000
RANGE_SIZE = 1_000


def _add_enum_extension(enum_el, registry: Registry, number: int):
    extends = enum_el.get("extends")
    if not extends:
        return  # not extending anything - e.g. a plain constant, alias, or feature bit reference

    group = registry.enums_groups.get(extends)
    if group is None:
        return  # TODO: think about whether this should ever happen

    offset  = enum_el.get("offset")
    bitpos  = enum_el.get("bitpos")
    value   = enum_el.get("value")
    op_dir  = enum_el.get("dir")
    name    = enum_el.get("name")
    type_   = enum_el.get("type")
    alias   = enum_el.get("alias")
    comment = enum_el.get("comment")

    if (not value) and offset:
        value = BASE_VALUE + (number - 1) * RANGE_SIZE + int(offset)
        if op_dir == "-": value = -value
        value = str(value)

    registry.enums_groups[extends].values.append(
        Enum_Value(
            name=name,
            value=value if value is not None else None,
            bitpos=int(bitpos) if bitpos is not None else None,
            type=C_Type(type_) if type_ is not None else None,
            alias=alias,
            comment=comment,
        )
    )


class Extension_Enum_Parser(Base_Parser):

    selection = "extensions"

    def parse(self, element, registry: Registry):
        for extension in element.findall("extension"):
            supported = extension.get("supported", "")
            if "vulkan" not in supported.split(","):
                continue

            number = int(extension.get("number"))

            for require in extension.findall("require"):
                for enum_el in require.findall("enum"):
                    _add_enum_extension(enum_el, registry, number)


class Feature_Enum_Parser(Base_Parser):
    # <feature> blocks (VK_VERSION_1_1, 1_2, 1_3, 1_4...) also extend
    # existing enum groups - mostly VkStructureType values for structs
    # that started as an extension and got promoted to core. Each such
    # <enum> carries its own `extnumber` instead of inheriting a parent
    # extension's `number` (there's no wrapping <extension> element here).

    selection = "feature"

    def parse(self, element, registry: Registry):
        api = element.get("api", "vulkan")
        if "vulkan" not in api.split(","):
            return  # vulkansc-only feature level (e.g. VKSC_VERSION_1_0), skip

        for require in element.findall("require"):
            for enum_el in require.findall("enum"):
                extnumber = enum_el.get("extnumber")
                number = int(extnumber) if extnumber else 0
                _add_enum_extension(enum_el, registry, number)