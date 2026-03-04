"""JSON Schema -> Pydantic BaseModel 동적 변환기.

DeepJSONEval의 각 인스턴스는 고유한 JSON Schema를 가지므로,
런타임에 동적으로 Pydantic 모델을 생성한다.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, create_model


def json_schema_to_pydantic(
    schema: dict,
    with_descriptions: bool = True,
    model_name: str | None = None,
) -> type[BaseModel]:
    """JSON Schema dict를 Pydantic BaseModel 클래스로 변환.

    Args:
        schema: JSON Schema dictionary
        with_descriptions: True면 Field(description=...) 포함, False면 제거
        model_name: 생성할 모델 이름 (None이면 schema title 사용)
    """
    name = model_name or schema.get("title", "DynamicModel")
    fields = _build_fields(schema, with_descriptions, counter=[0])
    return create_model(name, **fields)  # type: ignore[call-overload]


def _resolve_type(
    prop_schema: dict,
    with_descriptions: bool,
    counter: list[int],
) -> Any:
    """단일 프로퍼티의 타입을 해석."""
    typ = prop_schema.get("type", "string")

    # enum 처리
    if "enum" in prop_schema:
        vals = tuple(prop_schema["enum"])
        return Literal[vals]  # type: ignore[valid-type]

    if typ == "string":
        return str
    elif typ == "number":
        return float
    elif typ == "integer":
        return int
    elif typ == "boolean":
        return bool
    elif typ == "array":
        items = prop_schema.get("items", {})
        item_type = _resolve_type(items, with_descriptions, counter)
        # items가 object면 이미 BaseModel 클래스
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return list[item_type]
        return list[item_type]
    elif typ == "object":
        # nested object -> 재귀적으로 새 BaseModel 생성
        counter[0] += 1
        nested_name = prop_schema.get("title", f"Nested_{counter[0]}")
        nested_fields = _build_fields(prop_schema, with_descriptions, counter)
        if not nested_fields:
            return dict[str, Any]
        return create_model(nested_name, **nested_fields)  # type: ignore[call-overload]
    else:
        return str


def _get_default(typ: Any) -> Any:
    """타입에 맞는 기본값 반환."""
    if typ is str:
        return ""
    elif typ is int:
        return 0
    elif typ is float:
        return 0.0
    elif typ is bool:
        return False
    # Literal
    origin = getattr(typ, "__origin__", None)
    if origin is Literal:
        return get_args(typ)[0]
    # list
    if origin is list:
        return []
    return None


def _build_fields(
    schema: dict,
    with_descriptions: bool,
    counter: list[int],
) -> dict[str, Any]:
    """schema의 properties를 Pydantic create_model용 필드 dict로 변환."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        resolved = _resolve_type(prop_schema, with_descriptions, counter)
        desc = prop_schema.get("description", "")
        is_required = prop_name in required

        # BaseModel 서브클래스는 default_factory 사용
        is_model = isinstance(resolved, type) and issubclass(resolved, BaseModel)
        # list 타입인 경우
        origin = getattr(resolved, "__origin__", None)
        is_list = origin is list

        if with_descriptions and desc:
            if is_model:
                field = (resolved, Field(default_factory=resolved, description=desc))
            elif is_list:
                field = (resolved, Field(default_factory=list, description=desc))
            else:
                default = ... if is_required else _get_default(resolved)
                field = (resolved, Field(default=default, description=desc))
        else:
            if is_model:
                field = (resolved, Field(default_factory=resolved))
            elif is_list:
                field = (resolved, Field(default_factory=list))
            else:
                default = ... if is_required else _get_default(resolved)
                field = (resolved, Field(default=default))

        fields[prop_name] = field

    return fields
