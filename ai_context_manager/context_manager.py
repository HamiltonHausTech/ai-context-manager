
import logging

from typing import Dict, List, Optional, Union, Any

from ai_context_manager.feedback import Feedback

from ai_context_manager.components import ContextComponent
from ai_context_manager.components.task_summary import TaskSummaryComponent
from ai_context_manager.summarizers import Summarizer, NaiveSummarizer
from ai_context_manager.store import MemoryStore
from ai_context_manager.utils import estimate_tokens, component_from_dict

class ContextManager:
    def __init__(self, feedback: Optional[Feedback] = None,
                 memory_store: Optional[MemoryStore] = None,
                 config: Optional[Dict] = None,
                 summarizer: Optional[Summarizer] = None):
        self.components: Dict[str, ContextComponent] = {}
        self.feedback = feedback
        self.memory_store = memory_store
        self.config = config or {}
        self.summarizer = summarizer or NaiveSummarizer()

        
        if self.memory_store:
            self.load_from_memory_store()

    def load_from_memory_store(self):
        if not self.memory_store:
            return
        try:
            raw_components = self.memory_store.load_all()
            for raw in raw_components:
                try:
                    comp_id = raw.get("id")
                    if not comp_id:
                        raise ValueError("Missing component ID")
                    comp = component_from_dict(comp_id,raw)
                    self.register_component(comp)
                except Exception as e:
                    logging.warning(f"Failed to load component {raw.get('id')}: {e}")
        except Exception as e:
            logging.error(f"Error loading memory store: {e}")

    def save_component_to_memory(self, component: ContextComponent):
        if not self.memory_store:
            return
        try:
            comp_data = {
                "id": component.id,
                "tags": component.tags,
                "content": component.get_content(),
                "type": component.__class__.__name__,
            }
            self.memory_store.save_component(comp_data)
        except Exception as e:
            logging.error(f"Failed to save component {component.id}: {e}")

    def register_component(self, component: ContextComponent):
        """Register a component with proper validation and error handling."""
        if not component or not hasattr(component, 'id'):
            raise ValueError("Component must have a valid ID")
        
        if not component.id:
            raise ValueError("Component ID cannot be empty")
        
        if component.id in self.components:
            logging.warning(f"Component with ID '{component.id}' already registered. Skipping.")
            return
        
        try:
            self.components[component.id] = component
            if self.memory_store:
                self.save_component_to_memory(component)
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
        include_tags: Optional[List[str]] = None,
        component_types: Optional[List[str]] = None,
        summarize_if_needed: bool = False,
        token_budget: Optional[int] = None,
        return_metadata: bool = False,
        dry_run: bool = False
    ) -> Union[str, List[Dict[str, Any]], None]:
        """Get context with comprehensive error handling and validation."""
        try:
            # Input validation
            if token_budget is not None and token_budget <= 0:
                raise ValueError("Token budget must be positive")
            
            if include_tags is not None and not isinstance(include_tags, list):
                raise ValueError("include_tags must be a list")
            
            if component_types is not None and not isinstance(component_types, list):
                raise ValueError("component_types must be a list")
            
            components = list(self.components.values())
            
            if not components:
                logging.warning("No components available for context generation")
                return "" if not return_metadata else []

            # Filter components
            if include_tags:
                components = [c for c in components if c.matches_tags(include_tags)]
            if component_types:
                components = [c for c in components if c.__class__.__name__ in component_types]
            
            if not components:
                logging.warning("No components match the specified filters")
                return "" if not return_metadata else []

            def get_effective_score(comp: ContextComponent) -> float:
                try:
                    if self.feedback:
                        id_score = self.feedback.get_average_score(comp.id)
                        type_score = self.feedback.get_average_score_by_type(comp.__class__.__name__)
                        base_score = comp.score() if hasattr(comp, "score") else 0.0
            
                        # Tune the weightings here as needed
                        return base_score + (id_score * 0.7) + (type_score * 0.3)
                    return comp.score() if hasattr(comp, "score") else 0.0
                except Exception as e:
                    logging.warning(f"Error calculating score for component {comp.id}: {e}")
                    return 0.0

            components.sort(key=get_effective_score, reverse=True)

            context_parts = []
            used_tokens = 0
            included_count = 0
            
            for comp in components:
                try:
                    content = comp.get_content()
                    token_count = estimate_tokens(content)
                    was_summarized = False

                    if token_budget and used_tokens + token_count > token_budget:
                        if summarize_if_needed and self.summarizer:
                            try:
                                content = self.summarizer.summarize(
                                    comp.get_content(), token_budget - used_tokens
                                )                    
                                new_token_count = estimate_tokens(content)
                                if used_tokens + new_token_count > token_budget:
                                    continue
                                token_count = new_token_count
                                was_summarized = True
                            except Exception as e:
                                logging.warning(f"Failed to summarize component {comp.id}: {e}")
                                continue
                        else:
                            continue            
                    
                    score = get_effective_score(comp)

                    if dry_run:
                        print(comp.render_preview(score,token_count,was_summarized))
                        included_count += 1
                        used_tokens += token_count
                        continue

                    used_tokens += token_count
                    if return_metadata:
                        context_parts.append({
                            "id": comp.id,
                            "type": comp.__class__.__name__,
                            "tags": comp.tags,
                            "score": score,
                            "tokens": token_count,
                            "content": content
                        })
                    else:
                        context_parts.append(content)
                        
                except Exception as e:
                    logging.error(f"Error processing component {comp.id}: {e}")
                    continue
            
            if dry_run:
                print(f"=== Dry Run Complete: {included_count} components would have been included ===")
                return
                
            return context_parts if return_metadata else "\n\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"Error in get_context: {e}")
            raise