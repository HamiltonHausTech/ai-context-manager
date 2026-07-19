
import logging

from typing import Callable, Dict, List, Optional, Union, Any

from ai_context_manager.feedback import Feedback

from ai_context_manager.components import ContextComponent
from ai_context_manager.summarizers import Summarizer, NaiveSummarizer
from ai_context_manager.store import MemoryStore
from ai_context_manager.utils import component_from_dict, component_to_dict
from ai_context_manager.retrieval import RetrievalPipeline, RetrievalRequest, RetrievalResult

class ContextManager:
    def __init__(self, feedback: Optional[Feedback] = None,
                 memory_store: Optional[MemoryStore] = None,
                 config: Optional[Dict] = None,
                 summarizer: Optional[Summarizer] = None,
                 relevance_scorer: Optional[Callable[[str, ContextComponent], float]] = None):
        self.components: Dict[str, ContextComponent] = {}
        self.feedback = feedback
        self.memory_store = memory_store
        self.config = config or {}
        self.summarizer = summarizer or NaiveSummarizer()
        self.relevance_scorer = relevance_scorer

        
        if self.memory_store:
            self.load_from_memory_store()

    def load_from_memory_store(self):
        if not self.memory_store:
            return
        raw_components = self.memory_store.load_all()
        for raw in raw_components:
            try:
                comp_id = raw.get("id")
                if not comp_id:
                    raise ValueError("Missing component ID")
                comp = component_from_dict(comp_id,raw)
                self.register_component(comp, persist=False)
            except Exception as e:
                logging.warning(f"Failed to load component {raw.get('id')}: {e}")

    def save_component_to_memory(self, component: ContextComponent):
        if not self.memory_store:
            return
        comp_data = component_to_dict(component)
        self.memory_store.save_component(comp_data)

    def register_component(
        self,
        component: ContextComponent,
        on_duplicate: str = "skip",
        persist: bool = True,
    ):
        """Register a component with proper validation and error handling."""
        if not component or not hasattr(component, 'id'):
            raise ValueError("Component must have a valid ID")
        
        if not component.id:
            raise ValueError("Component ID cannot be empty")
        
        if on_duplicate not in ("skip", "replace", "error"):
            raise ValueError("on_duplicate must be 'skip', 'replace', or 'error'")

        if component.id in self.components:
            if on_duplicate == "skip":
                logging.warning(f"Component with ID '{component.id}' already registered. Skipping.")
                return
            if on_duplicate == "error":
                raise ValueError(f"Component with ID '{component.id}' is already registered")
        
        try:
            if self.memory_store and persist:
                self.save_component_to_memory(component)
            self.components[component.id] = component
            logging.debug(f"Successfully registered component: {component.id}")
        except Exception as e:
            logging.error(f"Failed to register component {component.id}: {e}")
            raise

    def remove_component(self, component_id: str):
        """Remove a component with proper error handling."""
        if not component_id:
            raise ValueError("Component ID cannot be empty")
        
        try:
            if component_id in self.components:
                del self.components[component_id]
                if self.memory_store:
                    self.memory_store.delete_component(component_id)
                logging.debug(f"Successfully removed component: {component_id}")
            else:
                logging.warning(f"Component {component_id} not found for removal")
        except Exception as e:
            logging.error(f"Failed to remove component {component_id}: {e}")
            raise

    def get_task_context(
        self,
        task_id: str,
        extra_tags: Optional[List[str]] = None,
        token_budget: int = 700
    ) -> Optional[str]:
        tags = ["task", "profile", "memory"]
        if extra_tags:
            tags.extend(extra_tags)
    
        result = self.get_context(
            include_tags=tags,
            task_id=task_id,
            summarize_if_needed=True,
            token_budget=token_budget,
            dry_run=False,
            return_metadata=False  # explicitly force str output
        )
        # Type checker guard
        if isinstance(result, str) or result is None:
            return result
        raise TypeError("Unexpected return type from get_context")

    def get_task_context_metadata(
        self,
        task_id: str,
        extra_tags: Optional[List[str]] = None,
        token_budget: int = 700
        ) -> Optional[List[Dict[str, Any]]]:
        tags = ["task", "profile", "memory"]
        if extra_tags:
            tags.extend(extra_tags)
    
        result = self.get_context(
            include_tags=tags,
            task_id=task_id,
            summarize_if_needed=True,
            token_budget=token_budget,
            dry_run=False,
            return_metadata=True
        )
    
        if isinstance(result, list):
            return result
        return None

    def get_context(
        self,
        query: Optional[str] = None,
        required_terms: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None,
        component_types: Optional[List[str]] = None,
        summarize_if_needed: bool = False,
        token_budget: Optional[int] = None,
        return_metadata: bool = False,
        dry_run: bool = False,
        tag_match_mode: str = "any",
        task_id: Optional[str] = None,
        include_inactive: bool = False,
        min_relevance: float = 0.0,
        deduplicate: bool = False,
        redundancy_threshold: float = 0.88,
        max_components: Optional[int] = None,
    ) -> Union[str, List[Dict[str, Any]], None]:
        """Compatibility wrapper over the explicit retrieval pipeline."""
        result = self.retrieve(
            RetrievalRequest(
                query=query,
                required_terms=required_terms,
                include_tags=include_tags,
                component_types=component_types,
                summarize_if_needed=summarize_if_needed,
                token_budget=token_budget,
                tag_match_mode=tag_match_mode,
                task_id=task_id,
                include_inactive=include_inactive,
                min_relevance=min_relevance,
                deduplicate=deduplicate,
                redundancy_threshold=redundancy_threshold,
                max_components=max_components,
            )
        )
        if dry_run:
            for item in result.items:
                print(
                    item.component.render_preview(
                        item.score, item.tokens, item.summarized
                    )
                )
            print(
                f"=== Dry Run Complete: {len(result.items)} components would have been included ==="
            )
            return result.metadata()
        return result.metadata() if return_metadata else result.context

    def _score_component(self, component: ContextComponent) -> float:
        base_score = component.score() if hasattr(component, "score") else 0.0
        if not self.feedback:
            return base_score
        id_score = self.feedback.get_average_score(component.id)
        type_score = self.feedback.get_average_score_by_type(
            component.__class__.__name__
        )
        return base_score + (id_score * 0.7) + (type_score * 0.3)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve context with decisions explaining every candidate outcome."""
        pipeline = RetrievalPipeline(
            self._score_component, self.summarizer, self.relevance_scorer
        )
        return pipeline.retrieve(list(self.components.values()), request)
