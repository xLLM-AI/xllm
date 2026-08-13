from __future__ import annotations

import re

from ...common.manifest import KernelAbi, KernelAbiParameter


def rename_entry_symbol(source: str, source_entry_symbol: str, entry_symbol: str) -> str:
    pattern = rf"\b{re.escape(source_entry_symbol)}\b"
    return re.sub(pattern, entry_symbol, source)


def rename_variant_internal_symbols(source: str, variant_key: str) -> str:
    symbol_names: set[str] = set()
    symbol_names.update(
        re.findall(
            r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
            source,
        )
    )
    symbol_names.update(re.findall(r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*_tiling)\s*\(", source))

    renamed_source = source
    for symbol_name in sorted(symbol_names, key=len, reverse=True):
        renamed_source = re.sub(
            rf"\b{re.escape(symbol_name)}\b",
            f"{symbol_name}__{variant_key}",
            renamed_source,
        )
    return renamed_source


def normalize_cpp_type(cpp_type: str) -> str:
    normalized = re.sub(r"\s+", " ", cpp_type).strip()
    normalized = re.sub(r"\s*([*&]+)\s*", r"\1", normalized)
    return normalized


def parse_kernel_abi(source: str, entry_symbol: str) -> KernelAbi:
    pattern = re.compile(
        rf'extern\s+"C"\s+'
        rf"(?P<return_type>[^(){{}};]+?)\s+"
        rf"{re.escape(entry_symbol)}\s*\("
        r"(?P<params>[^)]*)\)\s*\{",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Failed to parse exported entry ABI for symbol {entry_symbol!r}")

    return_type = normalize_cpp_type(match.group("return_type"))
    params_text = match.group("params").strip()
    parameters: list[KernelAbiParameter] = []
    if params_text and params_text != "void":
        for param in (part.strip() for part in params_text.split(",")):
            parsed = re.match(
                r"(?P<type>.+?[\*&]?)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)$",
                param,
            )
            if parsed is None:
                raise ValueError(f"Failed to parse kernel ABI parameter {param!r} for symbol {entry_symbol!r}")
            parameters.append(
                KernelAbiParameter(
                    cpp_type=normalize_cpp_type(parsed.group("type")),
                    name=parsed.group("name"),
                )
            )

    return KernelAbi(return_type=return_type, parameters=parameters)
