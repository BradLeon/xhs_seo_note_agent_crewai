"""JSON guardrail utilities for CrewAI tasks.

Provides robust JSON extraction from LLM outputs that commonly include:
- Markdown code blocks (```json ... ```)
- Prefix text ("Final Answer:", "Here is the JSON:", etc.)
- Trailing text after JSON
- Duplicate JSON objects (LLM repeating output)
"""

from typing import Any, Tuple
import json
import re

from crewai import TaskOutput


def format_json_output(result: TaskOutput) -> Tuple[bool, Any]:
    """Extract and format JSON from agent output.

    This is a guardrail function for CrewAI tasks that:
    1. Extracts JSON from markdown code blocks
    2. Uses brace-counting to extract FIRST complete JSON object
    3. Validates and re-serializes to ensure clean output

    Args:
        result: TaskOutput from CrewAI agent

    Returns:
        Tuple of (success: bool, result: str | error_message)
        - If success: (True, clean_json_string)
        - If failure: (False, error_message)
    """
    raw = result.raw.strip()

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
    if json_match:
        raw = json_match.group(1).strip()

    # Find first { to start extraction
    start = raw.find('{')
    if start == -1:
        return (False, f"No JSON object found. Raw output: {raw[:200]}")

    # Extract FIRST complete JSON object by counting braces
    # This handles: "Final Answer: {...}Final Answer: {...}" → extracts first {...}
    brace_count = 0
    in_string = False
    escape_next = False
    end = start

    for i, char in enumerate(raw[start:], start=start):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if brace_count != 0:
        # Brace mismatch, fall back to simple extraction
        end = raw.rfind('}')

    if end > start:
        raw = raw[start:end + 1]

    # Validate and parse JSON - return re-serialized to ensure clean output
    try:
        parsed = json.loads(raw)
        # Return JSON string (not dict) - CrewAI expects string for output_pydantic
        return (True, json.dumps(parsed, ensure_ascii=False))
    except json.JSONDecodeError as e:
        return (False, f"Invalid JSON after formatting: {str(e)[:100]}. Extracted: {raw[:300]}")


def fix_common_type_errors(data: dict, schema_hints: dict) -> dict:
    """Fix common LLM type errors based on schema hints.

    LLMs often output strings instead of arrays, or arrays instead of strings.
    This function attempts to fix these common issues.

    Args:
        data: Parsed JSON dict from LLM output
        schema_hints: Dict mapping field paths to expected types
            e.g., {"text_features.value_hierarchy": list, "visual_features.color_scheme": str}

    Returns:
        Fixed dict with corrected types
    """
    def get_nested(obj, path):
        """Get nested value by dot-separated path."""
        keys = path.split('.')
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return None
        return obj

    def set_nested(obj, path, value):
        """Set nested value by dot-separated path."""
        keys = path.split('.')
        for key in keys[:-1]:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value

    for path, expected_type in schema_hints.items():
        current_value = get_nested(data, path)
        if current_value is None:
            continue

        if expected_type == list and isinstance(current_value, str):
            # Convert string to list by splitting on common delimiters
            if '\n' in current_value:
                # Numbered list like "1. item\n2. item"
                items = [line.strip() for line in current_value.split('\n') if line.strip()]
                # Remove numbering prefixes
                items = [re.sub(r'^\d+\.\s*', '', item) for item in items]
                set_nested(data, path, items)
            elif '、' in current_value:
                # Chinese comma-separated
                set_nested(data, path, [item.strip() for item in current_value.split('、')])
            elif ',' in current_value:
                # English comma-separated
                set_nested(data, path, [item.strip() for item in current_value.split(',')])
            else:
                # Single item
                set_nested(data, path, [current_value])

        elif expected_type == str and isinstance(current_value, list):
            # Convert list to string
            set_nested(data, path, ', '.join(str(item) for item in current_value))

    return data
