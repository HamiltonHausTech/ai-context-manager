from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional

from ai_context_manager.components import LongTermMemoryComponent, TaskSummaryComponent
from ai_context_manager.consolidation import ConsolidationEngine
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.embeddings import stable_content_hash
from ai_context_manager.memory import MemoryLifecycle
from ai_context_manager.retrieval import RetrievalRequest
from ai_context_manager.store.sqlite_memory import SQLiteMemoryStore

from bourbon_research.models import Project
from bourbon_research.providers import (
    ResearchModel,
    SearchProvider,
    SourceFetcher,
    WikipediaSearchProvider,
    default_research_model,
    default_search_provider,
)
from bourbon_research.repository import ResearchRepository


class ResearchDesk:
    def __init__(
        self,
        workspace: str = ".bourbon-research",
        search_provider: Optional[SearchProvider] = None,
        fetcher: Optional[SourceFetcher] = None,
        model: Optional[ResearchModel] = None,
    ):
        self.repository = ResearchRepository(workspace)
        self.search_provider = search_provider or default_search_provider()
        self.fetcher = fetcher or SourceFetcher()
        self.model = model or default_research_model()

    def create_project(self, subject: str, objective: Optional[str] = None) -> Project:
        return self.repository.create_project(subject, objective)

    def project(self, slug: Optional[str] = None) -> Project:
        return self.repository.get_project(slug)

    def _context(self, project: Project) -> ContextManager:
        memory_path = self.repository.workspace / "projects" / project.slug / "memory.db"
        return ContextManager(memory_store=SQLiteMemoryStore(str(memory_path)))

    def plan(self, slug: Optional[str] = None) -> List[str]:
        project = self.project(slug)
        session = self.repository.start_session(project.id, "research plan")
        questions = self.model.plan(project.subject, project.objective)
        self.repository.replace_questions(project.id, questions)
        for question in questions:
            self.repository.event(session, "created", "question", question, question)
        self.repository.finish_session(session, f"Created {len(questions)} research questions")
        return questions

    def discover(
        self, slug: Optional[str] = None, max_sources: int = 10
    ) -> List[Dict]:
        project = self.project(slug)
        questions = self.repository.questions(project.id)
        if not questions:
            self.plan(project.slug)
            questions = self.repository.questions(project.id)
        discovered = []
        seen = set()
        subject_terms = re.findall(r"[a-z0-9]+", project.subject.lower())
        generic_terms = {
            "history", "research", "study", "overview", "whiskey", "whisky",
            "wine", "beer", "spirit", "spirits",
        }
        anchors = [term for term in subject_terms if term not in generic_terms]
        if not anchors:
            anchors = subject_terms

        def relevant(result, question_keywords=None) -> bool:
            if not isinstance(self.search_provider, WikipediaSearchProvider):
                return True
            title = result.title.lower()
            haystack = f"{title} {result.snippet} {result.url}".lower()
            anchor_in_title = any(anchor in title for anchor in anchors)
            anchor_anywhere = any(anchor in haystack for anchor in anchors)
            if question_keywords is None:
                return anchor_anywhere
            keyword_in_title = any(keyword in title for keyword in question_keywords)
            return anchor_in_title or (anchor_anywhere and keyword_in_title)

        per_question = max(2, max_sources // max(1, len(questions)) + 1)
        for result in self.search_provider.search(project.subject, per_question):
            if (
                not relevant(result)
                or result.url in seen
                or self.repository.source_exists(project.id, result.url)
            ):
                continue
            seen.add(result.url)
            discovered.append({"question": "Subject overview", "result": result})
            if len(discovered) >= max_sources:
                return discovered
        # A dry run must not turn a broad plan into dozens of API calls merely
        # because relevance filtering returns fewer than max_sources results.
        question_query_budget = max(1, max_sources)
        for question in questions[:question_query_budget]:
            stop_words = {
                "what", "which", "how", "the", "and", "are", "is", "to", "of",
                "for", "with", "about", "have", "has", "been", "does", "from",
                "understanding", "reliable", "documentary", "essential",
            }
            question_terms = [
                word for word in re.findall(r"[a-z0-9]+", question["text"].lower())
                if word not in stop_words and word not in subject_terms
            ]
            subject_query = f'"{project.subject}"' if isinstance(
                self.search_provider, WikipediaSearchProvider
            ) else project.subject
            query = " ".join([subject_query] + question_terms[:5])
            for result in self.search_provider.search(query, per_question):
                if (
                    not relevant(result, question_terms[:5])
                    or result.url in seen
                    or self.repository.source_exists(project.id, result.url)
                ):
                    continue
                seen.add(result.url)
                discovered.append({"question": question["text"], "result": result})
                if len(discovered) >= max_sources:
                    return discovered
        return discovered

    def run(
        self,
        slug: Optional[str] = None,
        max_sources: int = 10,
        dry_run: bool = False,
        source_policy: str = "exclude-community",
    ) -> Dict:
        project = self.project(slug)
        if dry_run:
            discovered = self.discover(project.slug, max_sources=max_sources)
            return {
                "project": project.slug,
                "discovered": [
                    {"url": item["result"].url, "title": item["result"].title,
                     "question": item["question"]}
                    for item in discovered
                ],
                "stored_sources": 0,
                "stored_claims": 0,
            }

        pending = self.repository.sources_without_claims(project.id)[:max_sources]
        remaining = max(0, max_sources - len(pending))
        discovered = self.discover(project.slug, max_sources=remaining) if remaining else []
        session = self.repository.start_session(project.id, "research run")
        manager = self._context(project)
        stored_sources = stored_claims = rejected = extraction_failures = 0
        try:
            work = [
                {
                    "source": self.repository.load_source_snapshot(row),
                    "source_id": row["id"],
                    "retry": True,
                }
                for row in pending
            ] + [{"result": item["result"], "retry": False} for item in discovered]
            for item in work:
                source = item.get("source")
                source_id = item.get("source_id")
                if not item["retry"]:
                    result = item["result"]
                    try:
                        source = self.fetcher.fetch(result)
                        if source_policy == "authoritative" and source.source_class not in {
                            "primary_authority", "strong_secondary"
                        }:
                            raise ValueError(
                                f"Source class {source.source_class} rejected by authoritative policy"
                            )
                        if source_policy == "exclude-community" and source.source_class in {
                            "community_or_folklore", "retail"
                        }:
                            raise ValueError(
                                f"Source class {source.source_class} rejected by policy"
                            )
                        source_id = self.repository.save_source(
                            project, source, stable_content_hash(source.text)
                        )
                        stored_sources += 1
                        self.repository.event(
                            session, "created", "source", source_id,
                            f"{source.source_class}: {source.title}",
                        )
                    except Exception as exc:
                        rejected += 1
                        self.repository.event(
                            session, "rejected", "source", result.url, str(exc)
                        )
                        continue

                source_memory_id = f"source-{source_id}"
                if source_memory_id not in manager.components:
                    manager.register_component(
                        TaskSummaryComponent(
                            source_memory_id,
                            source.title,
                            f"Read {source.url} ({source.source_class})",
                            tags=[project.slug, "source", source.source_class],
                        )
                    )
                try:
                    extracted = self.model.extract_claims(source)
                    for claim in extracted:
                        claim_id = self.repository.save_claim(project.id, source_id, claim)
                        memory_id = f"claim-{claim_id}"
                        self.repository.set_claim_memory(claim_id, memory_id)
                        if memory_id not in manager.components:
                            component = LongTermMemoryComponent(
                                memory_id,
                                claim.text,
                                source.url,
                                datetime.now(timezone.utc).isoformat(),
                                score=claim.confidence,
                                tags=[project.slug, "claim", claim.evidence_type],
                            )
                            component.set_memory_lifecycle(
                                MemoryLifecycle(
                                    kind="durable_fact",
                                    provenance_ids=[source_memory_id],
                                    confidence=claim.confidence,
                                )
                            )
                            manager.register_component(component)
                            stored_claims += 1
                            self.repository.event(
                                session, "created", "claim", claim_id, claim.text
                            )
                    self.repository.set_source_status(
                        source_id, "processed" if extracted else "no_claims"
                    )
                except Exception as exc:
                    extraction_failures += 1
                    self.repository.set_source_status(source_id, "extraction_failed")
                    self.repository.event(
                        session, "rejected", "extraction", source_id, str(exc)
                    )

            claims = self.repository.claims(project.id)
            engine = ConsolidationEngine(manager)
            for left_id, right_id in self.model.find_contradictions(claims):
                by_id = {claim["id"]: claim for claim in claims}
                if left_id not in by_id or right_id not in by_id:
                    continue
                left, right = by_id[left_id], by_id[right_id]
                self.repository.add_contradiction(project.id, left_id, right_id)
                if left.get("memory_id") and right.get("memory_id"):
                    engine.record_contradiction(left["memory_id"], right["memory_id"])
                    self.repository.event(
                        session, "detected", "contradiction",
                        f"{left_id}:{right_id}", "Claims require resolution",
                    )
            summary = (
                f"Stored {stored_sources} sources and {stored_claims} claims; "
                f"rejected {rejected} sources; {extraction_failures} extraction failures"
            )
            self.repository.finish_session(session, summary)
            return {
                "project": project.slug,
                "discovered": len(discovered),
                "stored_sources": stored_sources,
                "stored_claims": stored_claims,
                "rejected_sources": rejected,
                "retried_sources": len(pending),
                "extraction_failures": extraction_failures,
            }
        finally:
            if hasattr(manager.memory_store, "close"):
                manager.memory_store.close()

    def consolidate(self, slug: Optional[str] = None) -> str:
        project = self.project(slug)
        manager = self._context(project)
        try:
            claim_ids = [
                component.id for component in manager.components.values()
                if "claim" in component.tags and component.memory.is_active()
            ]
            if not claim_ids:
                raise ValueError("No active claims are available to consolidate")
            content = "\n".join(
                f"- {manager.components[claim_id].get_content()}" for claim_id in claim_ids
            )
            summary_id = f"research-summary-{len([key for key in manager.components if key.startswith('research-summary-')]) + 1}"
            ConsolidationEngine(manager).derive(
                summary_id,
                content,
                claim_ids,
                derivation="research synthesis",
                confidence=min(manager.components[item].memory.confidence for item in claim_ids),
                tags=[project.slug, "memory", "derived", "research-summary"],
            )
            return summary_id
        finally:
            manager.memory_store.close()

    def status(self, slug: Optional[str] = None) -> Dict:
        project = self.project(slug)
        questions = self.repository.questions(project.id)
        sources = self.repository.sources(project.id)
        claims = self.repository.claims(project.id)
        contradictions = self.repository.contradictions(project.id)
        return {
            "project": project.slug,
            "subject": project.subject,
            "questions": len(questions),
            "open_questions": sum(item["status"] == "open" for item in questions),
            "sources": len(sources),
            "source_classes": dict(Counter(item["source_class"] for item in sources)),
            "claims": len(claims),
            "evidence_types": dict(Counter(item["evidence_type"] for item in claims)),
            "open_contradictions": sum(item["status"] == "open" for item in contradictions),
        }

    def memory_trace(
        self,
        slug: Optional[str] = None,
        token_budget: int = 1000,
        query: Optional[str] = None,
    ) -> Dict:
        project = self.project(slug)
        manager = self._context(project)
        try:
            active_query = query or f"{project.subject}. {project.objective}"
            required_terms = [
                term for term in re.findall(r"[a-z0-9]+", project.subject.lower())
                if term not in {"whiskey", "whisky", "research", "history"}
            ]
            result = manager.retrieve(
                RetrievalRequest(
                    query=active_query,
                    required_terms=required_terms,
                    include_tags=[project.slug],
                    tag_match_mode="all",
                    token_budget=token_budget,
                    summarize_if_needed=True,
                    min_relevance=0.10,
                    deduplicate=True,
                    max_components=20,
                )
            )
            return {
                "query": active_query,
                "context": result.context,
                "used_tokens": result.used_tokens,
                "decisions": [decision.__dict__ for decision in result.decisions],
            }
        finally:
            manager.memory_store.close()

    def report(self, slug: Optional[str] = None, output: Optional[str] = None) -> Path:
        project = self.project(slug)
        questions = self.repository.questions(project.id)
        sources = self.repository.sources(project.id)
        claims = self.repository.claims(project.id)
        source_by_id = {source["id"]: source for source in sources}
        lines = [
            f"# {project.subject}", "", project.objective, "",
            "## Research questions", "",
        ]
        lines.extend(f"- {question['text']}" for question in questions)
        lines.extend(["", "## Findings", ""])
        for claim in claims:
            lines.append(
                f"- **{claim['evidence_type']}** ({claim['confidence']:.0%}): "
                f"{claim['text']} [S{claim['source_id']}]"
            )
        lines.extend(["", "## Sources", ""])
        for source in sources:
            lines.append(
                f"- **[S{source['id']}]** [{source['title']}]({source['url']}) — "
                f"{source['source_class']}"
            )
        path = Path(output) if output else (
            self.repository.workspace / "projects" / project.slug / "report.md"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path.resolve()

    def close(self):
        self.repository.close()
