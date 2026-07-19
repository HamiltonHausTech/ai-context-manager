import os
import json
import tempfile
from typing import List, Dict, Optional
from ai_context_manager.store.base import MemoryStore
from ai_context_manager.store.errors import StorageReadError, StorageWriteError

class JSONMemoryStore(MemoryStore):
    def __init__(self, filepath="memory.json"):
        self.filepath = filepath
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                raise StorageReadError(f"Cannot read JSON memory store: {self.filepath}") from exc
            if not isinstance(self.data, list):
                raise StorageReadError(f"JSON memory store must contain a list: {self.filepath}")
        else:
            self.data = []

    def _save(self, data):
        directory = os.path.dirname(os.path.abspath(self.filepath))
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=directory, prefix=".memory-", suffix=".tmp", delete=False
            ) as temp_file:
                temp_path = temp_file.name
                json.dump(data, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.filepath)
        except (OSError, TypeError, ValueError) as exc:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise StorageWriteError(f"Cannot write JSON memory store: {self.filepath}") from exc

    def load_all(self) -> List[Dict]:
        return self.data

    def save_component(self, component: Dict) -> None:
        if not component.get("id"):
            raise ValueError("Stored component must have a non-empty ID")
        existing = [c for c in self.data if c["id"] == component["id"]]
        if existing:
            next_data = [c if c["id"] != component["id"] else component for c in self.data]
        else:
            next_data = self.data + [component]
        self._save(next_data)
        self.data = next_data

    def delete_component(self, component_id: str) -> None:
        next_data = [c for c in self.data if c["id"] != component_id]
        self._save(next_data)
        self.data = next_data

    def get_component(self, component_id: str) -> Optional[Dict]:
        for c in self.data:
            if c["id"] == component_id:
                return c
        return None
