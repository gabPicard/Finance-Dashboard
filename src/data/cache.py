"""In-memory price cache with configurable TTL.

Thread-safe using a :class:`threading.Lock`. Cache keys follow the format
``{ticker}:{start}:{end}``. Values are pandas DataFrames. Stale entries are
evicted lazily on every :meth:`PriceCache.get` call.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class _CacheEntry:
    """A single cached item with its expiry timestamp."""

    data: pd.DataFrame
    expires_at: float


class PriceCache:
    """Thread-safe in-memory cache for price DataFrames.

    Parameters
    ----------
    ttl:
        Time-to-live in seconds (default: 900 = 15 minutes).
    """

    def __init__(self, ttl: int = 900) -> None:
        """Initialise the cache with the given TTL."""
        self._ttl = ttl
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(ticker: str, start: str, end: str) -> str:
        """Build a canonical cache key from ticker and date range.

        Parameters
        ----------
        ticker:
            Asset ticker symbol.
        start:
            ISO-8601 start date string.
        end:
            ISO-8601 end date string.

        Returns
        -------
        str
            Key formatted as ``{ticker}:{start}:{end}``.
        """
        return f"{ticker}:{start}:{end}"

    def get(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Retrieve a cached DataFrame, or ``None`` if missing/expired.

        Parameters
        ----------
        ticker:
            Asset ticker symbol.
        start:
            ISO-8601 start date string.
        end:
            ISO-8601 end date string.

        Returns
        -------
        pd.DataFrame | None
            Cached data, or ``None`` when the key is absent or stale.
        """
        key = self.make_key(ticker, start, end)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            return entry.data.copy()

    def set(self, ticker: str, start: str, end: str, data: pd.DataFrame) -> None:
        """Store a DataFrame in the cache.

        Parameters
        ----------
        ticker:
            Asset ticker symbol.
        start:
            ISO-8601 start date string.
        end:
            ISO-8601 end date string.
        data:
            Price DataFrame to cache.
        """
        key = self.make_key(ticker, start, end)
        entry = _CacheEntry(
            data=data.copy(),
            expires_at=time.monotonic() + self._ttl,
        )
        with self._lock:
            self._store[key] = entry

    def invalidate(self, ticker: str, start: str, end: str) -> None:
        """Remove a specific entry from the cache.

        Parameters
        ----------
        ticker:
            Asset ticker symbol.
        start:
            ISO-8601 start date string.
        end:
            ISO-8601 end date string.
        """
        key = self.make_key(ticker, start, end)
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    def evict_expired(self) -> int:
        """Remove all stale entries and return the number evicted.

        Returns
        -------
        int
            Number of entries that were removed.
        """
        now = time.monotonic()
        with self._lock:
            stale = [k for k, v in self._store.items() if now > v.expires_at]
            for k in stale:
                del self._store[k]
        return len(stale)

    def __len__(self) -> int:
        """Return the current number of entries (including potentially stale ones)."""
        with self._lock:
            return len(self._store)


# Module-level singleton used by ``fetch_data.py``
default_cache: PriceCache = PriceCache(ttl=900)
