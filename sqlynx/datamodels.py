from __future__ import annotations

from pydantic import BaseModel


class SQLResult(BaseModel):
    columns: list[str]
    data: list[tuple[int | str | float]]
    metadata: dict
