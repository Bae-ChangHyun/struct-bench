"""DeepJSONEval용 동적 rich 프롬프트 생성기.

각 인스턴스마다 JSON Schema가 다르므로 description 정보를 추출하여
런타임에 상세 프롬프트를 생성한다.
"""

from __future__ import annotations


def generate_rich_prompt(schema_dict: dict) -> str:
    """JSON Schema의 description들을 추출하여 상세 프롬프트 생성."""
    lines = [
        "You are a precise data extraction assistant.",
        "Extract structured information from the given text into the specified JSON schema.",
        "Preserve original values as they appear in the text (numbers, dates, names, etc.).",
        "If a field is not present in the text, use null or the appropriate default value.",
        "",
        "Extract the following fields:",
    ]
    _collect_field_descriptions(schema_dict, lines, indent=0)
    return "\n".join(lines)


def _collect_field_descriptions(
    schema: dict,
    lines: list[str],
    indent: int,
) -> None:
    """프로퍼티를 재귀적으로 순회하며 필드 설명 수집."""
    properties = schema.get("properties", {})
    prefix = "  " * indent

    for prop_name, prop_schema in properties.items():
        desc = prop_schema.get("description", "")
        typ = prop_schema.get("type", "string")

        if "enum" in prop_schema:
            enum_vals = ", ".join(str(v) for v in prop_schema["enum"])
            lines.append(f"{prefix}- {prop_name}: {desc} (allowed values: {enum_vals})")
        elif typ == "object":
            lines.append(f"{prefix}- {prop_name}: {desc} (object with nested fields)")
            _collect_field_descriptions(prop_schema, lines, indent + 1)
        elif typ == "array":
            items = prop_schema.get("items", {})
            if items.get("type") == "object":
                lines.append(f"{prefix}- {prop_name}: {desc} (array of objects)")
                _collect_field_descriptions(items, lines, indent + 1)
            else:
                item_type = items.get("type", "string")
                lines.append(f"{prefix}- {prop_name}: {desc} (array of {item_type})")
        else:
            lines.append(f"{prefix}- {prop_name}: {desc} (type: {typ})")
