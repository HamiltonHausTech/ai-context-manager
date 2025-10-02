# ai_context_manager/store/json_store.py

from .base import FeedbackStore
from datetime import datetime
import json, os
from collections import defaultdict
from typing import List, Dict

class JSONFeedbackStore(FeedbackStore):
    def __init__(self, filepath="feedback.json"):
        self.filepath = filepath
        self.component_scores = defaultdict(list)
        self.type_scores = defaultdict(list)
        self.load()

    def add_feedback(self, component_id: str, score: float, component_type: str):
        entry = {"score": score, "timestamp": datetime.utcnow().isoformat()}
        self.component_scores[component_id].append(entry)
        self.type_scores[component_type].append(entry)
        self.save()

    def get_scores(self, component_id: str) -> List[Dict]:
        return self.component_scores.get(component_id, [])

    def get_scores_by_type(self, component_type: str) -> List[Dict]:
        return self.type_scores.get(component_type, [])

    def get_tracked_component_ids(self) -> List[str]:
        return list(self.component_scores.keys())
    
    def get_tracked_component_types(self) -> List[str]:
        return list(self.type_scores.keys())


    def save(self):
        with open(self.filepath, "w") as f:
            json.dump({
                "component_scores": self.component_scores,
                "type_scores": self.type_scores
            }, f, indent=2)

    def load(self):
        if not os.path.exists(self.filepath):
            return
        with open(self.filepath, "r") as f:
            data = json.load(f)

            def normalize(entries):
                return [
                    {"score": e, "timestamp": datetime.utcnow().isoformat()}
                    if isinstance(e, float) else e
                    for e in entries
                ]

            self.component_scores = defaultdict(list, {
                k: normalize(v) for k, v in data.get("component_scores", {}).items()
            })
            self.type_scores = defaultdict(list, {
                k: normalize(v) for k, v in data.get("type_scores", {}).items()
            })
