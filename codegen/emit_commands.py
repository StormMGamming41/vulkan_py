
from model import Registry
from resolve import resolve_c_type

def build_command_meta(registry: Registry) -> dict:
    meta = {}

    for cmd in registry.commands.values():
        if cmd.alias:
            continue  # handled in a second pass below

        info = {
            "returns_vkresult": cmd.return_type is not None and cmd.return_type.name == "VkResult",
        }

        params = cmd.params

        # Pattern 1: single output handle - non-const pointer to a Handle, no len (not an array)
        for i, p in enumerate(params):
            if (p.type.pointer_level == 1 and not p.type.const
                    and p.type.name in registry.handles and p.len is None):
                info["output_handle_index"] = i
                info["output_handle_type"] = p.type.name
                break

        # Pattern 2: count+array idiom - a param whose `len` names a sibling param
        for i, p in enumerate(params):
            if p.len and p.len != "null-terminated":
                for j, p2 in enumerate(params):
                    if p2.name == p.len:
                        info["count_index"] = j
                        info["array_index"] = i
                        info["array_element_type"] = p.type.name
                        break
                break

        if len(info) > 1 or info["returns_vkresult"]:
            meta[cmd.name] = info

    # aliases share their target's metadata
    for cmd in registry.commands.values():
        if cmd.alias and cmd.alias in meta:
            meta[cmd.name] = meta[cmd.alias]

    return meta

def emit_command_meta(registry: Registry) -> str:
    meta = build_command_meta(registry)
    return f"COMMAND_META = {meta!r}\n"

def emit_command_pointers(registry: Registry) -> str:
    lines = ["from ctypes import *", "from .types import *", "from .handles import *", "from .basetypes import *", ""]
    skipped = []

    skipped_names = set()

    for cmd in registry.commands.values():
        if cmd.alias:
            if cmd.alias in skipped_names:
                skipped.append(cmd.name)
                skipped_names.add(cmd.name)
                continue
            lines.append(f"PFN_{cmd.name} = PFN_{cmd.alias}")
            continue

        try:
            restype = resolve_c_type(cmd.return_type, registry)
            argtypes = [resolve_c_type(p.type, registry) for p in cmd.params]
        except ValueError:
            skipped.append(cmd.name)
            skipped_names.add(cmd.name)
            continue

        restype_str = restype if restype is not None else "None"
        args_str = ", ".join([restype_str] + argtypes)
        lines.append(f"PFN_{cmd.name} = CFUNCTYPE({args_str})")

    lines.append("")
    lines.append(emit_command_meta(registry))

    if skipped:
        print(f"Skipped {len(skipped)} commands (unresolvable types): {skipped}")

    return "\n".join(lines)