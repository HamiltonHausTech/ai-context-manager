from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class MemoryStore(ABC):
    @abstractmethod
    def load_all(self) -> List[Dict]: pass

    @abstractmethod
    def save_component(self, component: Dict) -> None: pass

    @abstractmethod
    def delete_component(self, component_id: str) -> None: pass

    @abstractmethod
    def get_component(self, component_id: str) -> Optional[Dict]: pass


class FeedbackStore(ABC):
    @abstractmethod
    def add_feedback(self, component_id: str, score: float, component_type: str): pass

    @abstractmethod
    def get_scores(self, component_id: str) -> List[Dict]: pass

    @abstractmethod
    def get_scores_by_type(self, component_type: str) -> List[Dict]: pass

    @abstractmethod
    def get_tracked_component_ids(self) -> List[str]: pass
    
    @abstractmethod
    def get_tracked_component_types(self) -> List[str]: pass
