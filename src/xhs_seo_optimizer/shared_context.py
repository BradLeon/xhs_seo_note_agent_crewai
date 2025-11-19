"""Shared execution context for CrewAI tools.

This module provides a thread-safe singleton to share complex data between
@before_kickoff and tools, working around CrewAI's limitation where LLM agents
cannot reliably pass large complex lists as tool parameters.

Pattern:
1. @before_kickoff stores serialized data in SharedContext
2. Tools fetch data from SharedContext instead of receiving as parameters
3. This prevents LLM hallucination of fake data
"""

import threading
from typing import Any, Dict, List, Optional


class SharedContext:
    """Thread-safe singleton for sharing data between crew hooks and tools."""

    _instance: Optional["SharedContext"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data: Dict[str, Any] = {}
        return cls._instance

    def set(self, key: str, value: Any) -> None:
        """Store data in shared context.

        Args:
            key: Data key
            value: Data value (can be any serializable type)
        """
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from shared context.

        Args:
            key: Data key
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        with self._lock:
            return self._data.get(key, default)

    def clear(self) -> None:
        """Clear all stored data (useful for testing)."""
        with self._lock:
            self._data.clear()

    def has(self, key: str) -> bool:
        """Check if key exists in context.

        Args:
            key: Data key

        Returns:
            True if key exists, False otherwise
        """
        with self._lock:
            return key in self._data


# Global singleton instance for easy import
shared_context = SharedContext()
