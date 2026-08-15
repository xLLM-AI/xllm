import tilelang
import tilelang.language as T
from typing import Any

from ....common.manifest import KernelAbi, KernelVariantManifest
from ....common.spec import DispatchField

DEFAULT_ASCEND_PASS_CONFIGS = {
    # Use raw pass-config strings to avoid hard dependency on
    # tilelang.PassConfigKey export timing/version.
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_auto_cv_combine": True,
}

DEFAULT_ASCEND_BISHENG_ARCH = "dav-2201"
ASCEND_VEC_CORE_NUM_PROPERTY_KEYS = (
    "vector_core_num",
    "aiv_core_num",
    "vec_core_num",
)


def detect_vec_core_num(default_vec_core_num: int = 48) -> int:
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            props = torch.npu.get_device_properties(torch.npu.current_device())
            for key in ASCEND_VEC_CORE_NUM_PROPERTY_KEYS:
                value = getattr(props, key, None)
                if isinstance(value, int) and value > 0:
                    return value
    except Exception:
        pass

    return default_vec_core_num


def _snake_to_pascal(name: str) -> str:
    parts = [part for part in name.split("_") if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _dispatch_field_suffix(name: str) -> str:
    parts = [part for part in name.split("_") if part]
    mapped = {"dtype": "DType"}
    return "".join(mapped.get(part, part[:1].upper() + part[1:]) for part in parts)


def _dtype_enum_suffix(dtype_name: str) -> str:
    common_suffixes = {
        "bf16": "BF16",
        "bfloat16": "BF16",
        "fp16": "Float16",
        "fp32": "Float32",
        "float16": "Float16",
        "float32": "Float32",
        "int8": "Int8",
        "int32": "Int32",
        "uint8": "UInt8",
    }
    if dtype_name in common_suffixes:
        return common_suffixes[dtype_name]
    return _snake_to_pascal(dtype_name)


def _dispatch_field_cpp_type(field: DispatchField) -> str:
    field_types = {
        "int32": "int32_t",
        "dtype": "TilelangDType",
    }
    return field_types[field.kind]


def _render_dispatch_value_literal(*, field: DispatchField, value: Any) -> str:
    if field.kind == "int32":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"Unsupported int32 dispatch value for {field.name!r}: {value!r}"
            )
        return str(value)
    if field.kind == "dtype":
        if not isinstance(value, str):
            raise TypeError(
                f"Unsupported dtype dispatch value for {field.name!r}: {value!r}"
            )
        dtype_suffix = _dtype_enum_suffix(value)
        return f"TilelangDType::k{dtype_suffix}"
    raise TypeError(
        f"Unsupported dispatch field kind {field.kind!r} for {field.name!r}"
    )


def _validate_dispatch_values(
    *, kernel_name: str, dispatch_schema: list[DispatchField], variant: KernelVariantManifest
) -> None:
    schema_names = [field.name for field in dispatch_schema]
    missing_keys = [name for name in schema_names if name not in variant.dispatch_values]
    extra_keys = [name for name in variant.dispatch_values if name not in schema_names]
    if missing_keys or extra_keys:
        raise ValueError(
            f"TileLang kernel family {kernel_name!r} variant "
            f"{variant.variant_key!r} has inconsistent dispatch values: "
            f"missing={missing_keys}, extra={extra_keys}"
        )


def render_family_variants_inc(
    *,
    kernel_name: str,
    dispatch_schema: list[DispatchField],
    variants: list[KernelVariantManifest],
) -> str:
    if not variants:
        return ""

    if not dispatch_schema:
        raise ValueError(
            f"TileLang kernel family {kernel_name!r} has empty dispatch schema"
        )

    macro_name = f"XLLM_TL_{kernel_name.upper()}_VARIANT"
    lines: list[str] = []

    for variant in variants:
        _validate_dispatch_values(
            kernel_name=kernel_name,
            dispatch_schema=dispatch_schema,
            variant=variant,
        )
        variant_args = [
            _render_dispatch_value_literal(
                field=field,
                value=variant.dispatch_values[field.name],
            )
            for field in dispatch_schema
        ]
        variant_args.append(f"\"{variant.variant_key}\"")
        variant_args.append(variant.entry_symbol)
        lines.append(f"{macro_name}({', '.join(variant_args)})")

    return "\n".join(lines) + "\n"


def render_family_registry_inc(
    *,
    kernel_name: str,
    dispatch_schema: list[DispatchField],
    kernel_abi: KernelAbi,
    variants: list[KernelVariantManifest],
) -> str:
    if not variants:
        return ""

    if not dispatch_schema:
        raise ValueError(
            f"TileLang kernel family {kernel_name!r} has empty dispatch schema"
        )

    family_prefix = _snake_to_pascal(kernel_name)
    specialization_type = f"{family_prefix}Specialization"
    kernel_fn_type = f"{family_prefix}KernelFn"
    registry_name = f"k{family_prefix}Registry"
    entry_type = f"KernelEntry<{specialization_type}, {kernel_fn_type}>"
    field_wrapper_types = {
        field.name: f"{family_prefix}{_dispatch_field_suffix(field.name)}"
        for field in dispatch_schema
    }

    symbol_declarations: list[str] = []
    registry_entries: list[str] = []
    for variant in variants:
        _validate_dispatch_values(
            kernel_name=kernel_name,
            dispatch_schema=dispatch_schema,
            variant=variant,
        )
        specialization_args = ", ".join(
            f"{field_wrapper_types[field.name]}{{"
            f"{_render_dispatch_value_literal(field=field, value=variant.dispatch_values[field.name])}"
            f"}}"
            for field in dispatch_schema
        )
        symbol_declarations.append(
            f'extern "C" function_type_t<{kernel_fn_type}> {variant.entry_symbol};'
        )
        registry_entries.append(
            f'    {entry_type}{{make_{kernel_name}_specialization({specialization_args}), '
            f'"{variant.variant_key}", &{variant.entry_symbol}}},'
        )

    struct_fields = [
        f"  {_dispatch_field_cpp_type(field)} {field.name};" for field in dispatch_schema
    ]
    equality_terms = [f"lhs.{field.name} == rhs.{field.name}" for field in dispatch_schema]
    builder_params = [
        f"{field_wrapper_types[field.name]} {field.name}" for field in dispatch_schema
    ]
    builder_values = ", ".join(f"{field.name}.value" for field in dispatch_schema)
    function_params = ", ".join(
        f"{parameter.cpp_type} {parameter.name}" for parameter in kernel_abi.parameters
    )

    lines = [
        f"struct {specialization_type} {{",
        *struct_fields,
        "};",
        "",
        f"constexpr bool operator==(const {specialization_type}& lhs,",
        f"                          const {specialization_type}& rhs) {{",
        "  return " + " && ".join(equality_terms) + ";",
        "}",
        "",
    ]
    for field in dispatch_schema:
        lines.extend(
            [
                f"struct {field_wrapper_types[field.name]} {{",
                f"  {_dispatch_field_cpp_type(field)} value;",
                "};",
                "",
            ]
        )
    lines.extend(
        [
            f"constexpr {specialization_type} make_{kernel_name}_specialization(",
            "    " + ", ".join(builder_params) + ") {",
            f"  return {specialization_type}{{{builder_values}}};",
            "}",
            "",
            f"using {kernel_fn_type} = {kernel_abi.return_type} (*)({function_params});",
            "",
            *symbol_declarations,
            "",
            f"constexpr std::array<{entry_type}, {len(variants)}> {registry_name}{{{{",
            *registry_entries,
            "}};",
            "",
            f"inline const {entry_type}* find_{kernel_name}_kernel_entry(",
            f"    const {specialization_type}& specialization) {{",
            f"  return find_kernel_entry({registry_name}, specialization);",
            "}",
            "",
            f"inline std::string available_{kernel_name}_variant_keys() {{",
            f"  return available_variant_keys({registry_name});",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pipeline sync macros for TileLang Ascend kernels with
# tl.ascend_auto_sync=False. These helpers keep kernel implementations small
# and centralize flag-direction naming.
# ---------------------------------------------------------------------------


@T.macro
def mte2_notify_v(event_id: T.int32):
    T.set_flag("mte2", "v", event_id)


@T.macro
def v_wait_mte2(event_id: T.int32):
    T.wait_flag("mte2", "v", event_id)


@T.macro
def v_notify_mte2(event_id: T.int32):
    T.set_flag("v", "mte2", event_id)


@T.macro
def mte2_wait_v(event_id: T.int32):
    T.wait_flag("v", "mte2", event_id)


@T.macro
def v_notify_mte3(event_id: T.int32):
    T.set_flag("v", "mte3", event_id)


@T.macro
def mte3_wait_v(event_id: T.int32):
    T.wait_flag("v", "mte3", event_id)


@T.macro
def mte3_notify_v(event_id: T.int32):
    T.set_flag("mte3", "v", event_id)


@T.macro
def v_wait_mte3(event_id: T.int32):
    T.wait_flag("mte3", "v", event_id)


@T.macro
def mte3_notify_mte2(event_id: T.int32):
    T.set_flag("mte3", "mte2", event_id)


@T.macro
def mte2_wait_mte3(event_id: T.int32):
    T.wait_flag("mte3", "mte2", event_id)


@T.macro
def mte2_notify_mte3(event_id: T.int32):
    T.set_flag("mte2", "mte3", event_id)


@T.macro
def mte3_wait_mte2(event_id: T.int32):
    T.wait_flag("mte2", "mte3", event_id)
