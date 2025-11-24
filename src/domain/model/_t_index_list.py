from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TIndexList:
    """Value object that validates and stores the t-index sequence."""

    raw_value: str
    value: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.raw_value, str):
            raise ValueError("t_index_list must be provided as a string.")

        try:
            parsed = json.loads(self.raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError("t_index_list must be a JSON array formatted string.") from exc

        if not isinstance(parsed, list) or not parsed:
            raise ValueError("t_index_list must be a non-empty list of integers.")

        validated: list[int] = []
        for idx, item in enumerate(parsed):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"t_index_list[{idx}] must be an int, got {type(item).__name__}.")
            validated.append(item)

        object.__setattr__(self, "value", tuple(validated))

    def as_list(self) -> list[int]:
        return list(self.value)
