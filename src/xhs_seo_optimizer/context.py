"""Shared context for CrewAI agents and tools.

This module provides a shared context mechanism to pass complex Python objects
between agents and tools without serialization limitations.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from xhs_seo_optimizer.models.note import Note

logger = logging.getLogger(__name__)


class CrewContext:
    """Shared context accessible to all tools in the crew.

    This solves the CrewAI limitation where complex Python objects
    cannot be directly passed to LLM agents. Instead, we store objects
    here and pass string references through the LLM.

    Architecture:
    1. Complex objects (e.g., List[Note]) are stored with a unique key
    2. Agents coordinate using string keys (serializable)
    3. Tools retrieve actual objects using the keys

    This maintains the principle: "agents coordinate, tools execute"
    while handling complex data structures efficiently.
    """

    def __init__(self):
        """Initialize the shared context."""
        self.data_cache: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        logger.info("CrewContext initialized")

    def store_notes(self, key: str, notes: List[Note]) -> str:
        """Store a list of notes and return the reference key.

        Args:
            key: Unique identifier for this note collection
            notes: List of Note objects to store

        Returns:
            The storage key for reference

        Raises:
            ValueError: If notes list is empty
        """
        if not notes:
            raise ValueError("Cannot store empty notes list")

        self.data_cache[key] = notes
        self.metadata[key] = {
            'type': 'notes_list',
            'count': len(notes),
            'stored_at': datetime.utcnow().isoformat(),
        }

        logger.info(f"Stored {len(notes)} notes with key: {key}")
        return key

    def get_notes(self, key: str) -> List[Note]:
        """Retrieve notes by reference key.

        Args:
            key: The reference key for the notes

        Returns:
            List of Note objects

        Raises:
            KeyError: If key doesn't exist
        """
        if key not in self.data_cache:
            raise KeyError(f"No notes found for key: {key}")

        notes = self.data_cache[key]
        logger.info(f"Retrieved {len(notes)} notes with key: {key}")
        return notes

    def store_note(self, key: str, note: Note) -> str:
        """Store a single note.

        Args:
            key: Unique identifier for this note
            note: Note object to store

        Returns:
            The storage key for reference
        """
        self.data_cache[key] = note
        self.metadata[key] = {
            'type': 'single_note',
            'note_id': note.note_id,
            'stored_at': datetime.utcnow().isoformat(),
        }

        logger.info(f"Stored note {note.note_id} with key: {key}")
        return key

    def get_note(self, key: str) -> Note:
        """Retrieve a single note by reference key.

        Args:
            key: The reference key for the note

        Returns:
            Note object

        Raises:
            KeyError: If key doesn't exist
        """
        if key not in self.data_cache:
            raise KeyError(f"No note found for key: {key}")

        note = self.data_cache[key]
        logger.info(f"Retrieved note {note.note_id} with key: {key}")
        return note

    def store_data(self, key: str, data: Any, data_type: str = "generic") -> str:
        """Store any data with a key.

        Args:
            key: Unique identifier for the data
            data: Any Python object to store
            data_type: Type description for metadata

        Returns:
            The storage key for reference
        """
        self.data_cache[key] = data
        self.metadata[key] = {
            'type': data_type,
            'stored_at': datetime.utcnow().isoformat(),
        }

        logger.info(f"Stored {data_type} data with key: {key}")
        return key

    def get_data(self, key: str) -> Any:
        """Retrieve any data by reference key.

        Args:
            key: The reference key for the data

        Returns:
            The stored data

        Raises:
            KeyError: If key doesn't exist
        """
        if key not in self.data_cache:
            raise KeyError(f"No data found for key: {key}")

        return self.data_cache[key]

    def exists(self, key: str) -> bool:
        """Check if a key exists in the context.

        Args:
            key: The reference key to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self.data_cache

    def list_keys(self) -> List[str]:
        """List all stored keys.

        Returns:
            List of all keys in the context
        """
        return list(self.data_cache.keys())

    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get metadata for a stored item.

        Args:
            key: The reference key

        Returns:
            Metadata dictionary or None if key doesn't exist
        """
        return self.metadata.get(key)

    def clear(self):
        """Clear all stored data."""
        count = len(self.data_cache)
        self.data_cache.clear()
        self.metadata.clear()
        logger.info(f"Cleared {count} items from context")

    def remove(self, key: str) -> bool:
        """Remove a specific item from context.

        Args:
            key: The reference key to remove

        Returns:
            True if removed, False if key didn't exist
        """
        if key in self.data_cache:
            del self.data_cache[key]
            if key in self.metadata:
                del self.metadata[key]
            logger.info(f"Removed key: {key}")
            return True
        return False


# Global instance for singleton pattern (optional)
_global_context: Optional[CrewContext] = None


def get_global_context() -> CrewContext:
    """Get or create the global CrewContext instance.

    Returns:
        The global CrewContext instance
    """
    global _global_context
    if _global_context is None:
        _global_context = CrewContext()
    return _global_context