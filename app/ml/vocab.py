from __future__ import annotations

from collections.abc import Iterable


class Vocab:
    """Maps a set of raw ids to contiguous embedding indices and back."""

    def __init__(self, ids: Iterable[int]):
        self.ids: list[int] = list(ids)
        self.index: dict[int, int] = {raw: i for i, raw in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def to_index(self, raw_id: int) -> int | None:
        return self.index.get(raw_id)

    def to_id(self, index: int) -> int:
        return self.ids[index]

    def __contains__(self, raw_id: int) -> bool:
        return raw_id in self.index
