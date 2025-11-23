from typing import Any


def drop_empty_strings(obj: Any) -> Any:
    """
    Recursively remove empty-string values from dicts/lists.
    Returns None in place of empty strings; callers should handle None appropriately.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            cleaned_value = drop_empty_strings(value)
            if cleaned_value is None:
                continue
            cleaned[key] = cleaned_value
        return cleaned
    if isinstance(obj, list):
        cleaned_list = []
        for value in obj:
            cleaned_value = drop_empty_strings(value)
            if cleaned_value is None:
                continue
            cleaned_list.append(cleaned_value)
        return cleaned_list
    if isinstance(obj, str):
        return obj if obj.strip() != "" else None
    return obj
