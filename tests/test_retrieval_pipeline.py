from ai_context_manager.components import ContextComponent, TaskSummaryComponent
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.retrieval import RetrievalRequest


def test_retrieval_result_explains_filters_and_inclusions():
    manager = ContextManager()
    manager.register_component(
        TaskSummaryComponent("wanted", "Wanted", "relevant", tags=["alpha", "beta"])
    )
    manager.register_component(
        TaskSummaryComponent("wrong-task", "Wrong", "other", tags=["alpha", "beta"])
    )
    manager.register_component(
        TaskSummaryComponent("wrong-tags", "Tags", "other", tags=["alpha"])
    )

    result = manager.retrieve(
        RetrievalRequest(
            include_tags=["alpha", "beta"],
            tag_match_mode="all",
            task_id="wanted",
        )
    )
    decisions = {decision.component_id: decision for decision in result.decisions}

    assert result.context == "Task: Wanted\nSummary: relevant"
    assert decisions["wanted"].reason == "included"
    assert decisions["wrong-task"].reason == "task_filter"
    assert decisions["wrong-tags"].reason == "tag_filter"


def test_retrieval_result_explains_budget_exclusion():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("large", "Large", "word " * 100))

    result = manager.retrieve(RetrievalRequest(token_budget=3))

    assert result.context == ""
    assert result.decisions[0].reason == "over_budget"
    assert result.decisions[0].tokens > 3


def test_retrieval_pipeline_accepts_authoritative_component_token_counter():
    from ai_context_manager.retrieval import RetrievalPipeline

    component = TaskSummaryComponent("authoritative", "Large", "word " * 100)
    request = RetrievalRequest(token_budget=1)
    pipeline = RetrievalPipeline(
        lambda _component: 1.0,
        token_counter=lambda _component, _content: 1,
    )

    result = pipeline.retrieve([component], request)

    assert result.used_tokens == 1
    assert result.items[0].tokens == 1
    assert result.decisions[0].tokens == 1


def test_injected_token_counter_receives_summary_and_drives_replacement_count():
    from ai_context_manager.retrieval import RetrievalPipeline

    class FixedSummarizer:
        def summarize(self, text, max_tokens):
            return "replacement summary"

    component = TaskSummaryComponent("summarized", "Large", "word " * 100)
    calls = []

    def counter(received_component, content):
        calls.append((received_component, content))
        return 100 if content == component.get_content() else 2

    pipeline = RetrievalPipeline(
        lambda _component: 1.0,
        summarizer=FixedSummarizer(),
        token_counter=counter,
    )
    result = pipeline.retrieve(
        [component], RetrievalRequest(token_budget=3, summarize_if_needed=True)
    )

    assert [content for _component, content in calls] == [
        component.get_content(),
        "replacement summary",
    ]
    assert all(received is component for received, _content in calls)
    assert result.items[0].content == "replacement summary"
    assert result.items[0].tokens == result.used_tokens == 2
    assert result.decisions[0].tokens == 2


def test_retrieval_result_reports_processing_errors():
    class BrokenComponent(ContextComponent):
        def load_content(self):
            raise RuntimeError("cannot render")

    manager = ContextManager()
    manager.register_component(BrokenComponent("broken"))

    result = manager.retrieve(RetrievalRequest())

    assert result.context == ""
    assert result.decisions[0].reason == "processing_error"


def test_retrieval_stages_can_be_inspected_independently():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("low", "Low", "one", score=1.0))
    manager.register_component(TaskSummaryComponent("high", "High", "two", score=2.0))
    request = RetrievalRequest()

    from ai_context_manager.retrieval import RetrievalPipeline

    pipeline = RetrievalPipeline(manager._score_component, manager.summarizer)
    candidates, decisions = pipeline.select_candidates(
        list(manager.components.values()), request
    )
    ranked = pipeline.rank_candidates(candidates)
    result = pipeline.pack_budget(ranked, request, decisions)

    assert [component.id for component in candidates] == ["low", "high"]
    assert [component.id for component, _score in ranked] == ["high", "low"]
    assert [item.component.id for item in result.items] == ["high", "low"]


def test_query_aware_retrieval_separates_relevance_from_importance():
    manager = ContextManager()
    manager.register_component(
        TaskSummaryComponent(
            "beer",
            "Beer definition",
            "Highly authoritative beer production rule",
            score=1.0,
        )
    )
    manager.register_component(
        TaskSummaryComponent(
            "bourbon",
            "Bourbon law",
            "Federal bourbon whiskey production requirements",
            score=0.5,
        )
    )

    result = manager.retrieve(
        RetrievalRequest(
            query="legal definition and production requirements for bourbon whiskey",
            required_terms=["bourbon"],
            min_relevance=0.1,
        )
    )
    decisions = {decision.component_id: decision for decision in result.decisions}

    assert [item.component.id for item in result.items] == ["bourbon"]
    assert decisions["beer"].reason == "required_term_miss"
    assert decisions["bourbon"].score_factors["relevance"] > 0
    assert decisions["bourbon"].score_factors["importance"] == 0.5


def test_query_aware_retrieval_can_suppress_redundant_context():
    manager = ContextManager()
    manager.register_component(
        TaskSummaryComponent(
            "one", "Bourbon", "Bourbon must use new charred oak barrels"
        )
    )
    manager.register_component(
        TaskSummaryComponent(
            "two", "Bourbon", "Bourbon must use new charred oak barrels"
        )
    )

    result = manager.retrieve(
        RetrievalRequest(query="bourbon barrel requirements", deduplicate=True)
    )

    assert len(result.items) == 1
    assert any(decision.reason == "redundant" for decision in result.decisions)


def test_legacy_retrieval_without_query_preserves_static_score_ordering():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("low", "Low", "bourbon", score=0.2))
    manager.register_component(TaskSummaryComponent("high", "High", "beer", score=0.9))

    result = manager.retrieve(RetrievalRequest())

    assert [item.component.id for item in result.items] == ["high", "low"]
    assert all("relevance" not in item.score_factors for item in result.items)


def test_custom_relevance_scorer_can_recover_conceptual_matches():
    def scorer(query, component):
        assert "bourbon" in query
        return 0.9 if "Bottled-in-Bond" in component.get_content() else 0.0

    manager = ContextManager(relevance_scorer=scorer)
    manager.register_component(
        TaskSummaryComponent(
            "bond", "Law", "The Bottled-in-Bond Act established spirits standards"
        )
    )
    manager.register_component(
        TaskSummaryComponent("beer", "Beer", "Malt beverage rule")
    )

    result = manager.retrieve(
        RetrievalRequest(query="bourbon legal history", min_relevance=0.2)
    )

    assert [item.component.id for item in result.items] == ["bond"]
    assert result.items[0].score_factors["relevance_method"] == "custom"


def test_query_is_supplied_to_compression():
    class RecordingSummarizer:
        def __init__(self):
            self.text = None

        def summarize(self, text, max_tokens):
            self.text = text
            return "bourbon summary"

    summarizer = RecordingSummarizer()
    manager = ContextManager(summarizer=summarizer)
    manager.register_component(
        TaskSummaryComponent("large", "Bourbon", "bourbon details " * 100)
    )

    result = manager.retrieve(
        RetrievalRequest(
            query="bourbon production", token_budget=20, summarize_if_needed=True
        )
    )

    assert result.items[0].summarized is True
    assert "Current task: bourbon production" in summarizer.text
