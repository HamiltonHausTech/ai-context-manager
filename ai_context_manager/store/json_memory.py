import os
import json
from typing import List, Dict, Optional
from ai_context_manager.store.base import MemoryStore

class JSONMemoryStore(MemoryStore):
    def __init__(self, filepath="memory.json"):
        self.filepath = filepath
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self.data = json.load(f)
        else:
            self.data = []

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)

    def load_all(self) -> List[Dict]:
        return self.data

    def save_component(self, component: Dict) -> None:
        existing = [c for c in self.data if c["id"] == component["id"]]
        if existing:
            self.data = [c if c["id"] != component["id"] else component for c in self.data]
        else:
            self.data.append(component)
        self._save()

    def delete_component(self, component_id: str) -> None:
        self.data = [c for c in self.data if c["id"] != component_id]
        self._save()

    def get_component(self, component_id: str) -> Optional[Dict]:
        for c in self.data:
            if c["id"] == component_id:
                return c
        return None
