"""Small in-process LRU + TTL cache. One process and one event loop, so no locking. With several
replicas the same interface would sit in front of Redis.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

_MISSING: Any = object()


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": self.entries,
            "hit_rate": round(self.hit_rate, 3),
        }


class TTLCache[K: Hashable, V]:
    def __init__(
        self,
        *,
        max_entries: int = 2000,
        ttl_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        sweep_every: int = 64,
    ) -> None:
        self._max = max(1, max_entries)
        self._ttl = ttl_s
        self._clock = clock
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        # expired entries are otherwise dropped only when read; sweeping every `sweep_every`
        # writes keeps memory tied to live entries without a background task
        self._sweep_every = max(1, sweep_every)
        self._sets_since_sweep = 0

    def get(self, key: K, default: V | Any = _MISSING) -> V | Any:
        item = self._data.get(key)
        if item is None:
            self._misses += 1
            return default
        expires, value = item
        if expires <= self._clock():
            del self._data[key]
            self._misses += 1
            return default
        self._data.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: K, value: V) -> None:
        self._data[key] = (self._clock() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)
        self._sets_since_sweep += 1
        if self._sets_since_sweep >= self._sweep_every:
            self.purge_expired()

    def purge_expired(self) -> int:
        """Drop every expired entry now. Returns how many were removed."""
        self._sets_since_sweep = 0
        now = self._clock()
        doomed = [k for k, (expires, _) in self._data.items() if expires <= now]
        for k in doomed:
            del self._data[k]
        return len(doomed)

    def invalidate(self, key: K) -> None:
        self._data.pop(key, None)

    def invalidate_where(self, predicate: Callable[[K], bool]) -> int:
        doomed = [k for k in self._data if predicate(k)]
        for k in doomed:
            del self._data[k]
        return len(doomed)

    def clear(self) -> None:
        self._data.clear()

    def keys(self) -> list[K]:
        """A snapshot of the current keys (expired entries included until touched)."""
        return list(self._data)

    def __contains__(self, key: K) -> bool:
        return self.get(key) is not _MISSING

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        # an empty cache is still a cache: `if cache:` must not depend on len()
        return True

    @property
    def stats(self) -> CacheStats:
        return CacheStats(hits=self._hits, misses=self._misses, entries=len(self._data))


MISSING = _MISSING
