from bourbon_research.cli import main
import bourbon_research.cli as research_cli
from bourbon_research.models import ExtractedClaim, FetchedSource, SearchResult
from bourbon_research.providers import (
    OpenAIResearchModel,
    OpenAIWebSearchProvider,
    ResearchModel,
    SearchProvider,
    WikipediaSearchProvider,
    classify_source,
)
from bourbon_research.service import ResearchDesk


class FakeSearch(SearchProvider):
    def search(self, query, limit=10):
        return [
            SearchResult("https://example.gov/history", "Government history"),
            SearchResult("https://museum.example.edu/whiskey", "Museum history"),
        ][:limit]


class FakeFetcher:
    def fetch(self, result):
        if "example.gov" in result.url:
            text = "Bourbon was formally described in this authoritative record. " * 5
            source_class = "primary_authority"
        else:
            text = "The museum states bourbon was not created by a single inventor. " * 5
            source_class = "strong_secondary"
        return FetchedSource(
            result.url, result.title, text, result.url.split("/")[2], source_class
        )


class FakeModel(ResearchModel):
    def plan(self, subject, objective):
        return ["What is documented?", "Which origin stories conflict?"]

    def extract_claims(self, source):
        if source.source_class == "primary_authority":
            return [
                ExtractedClaim(
                    "The authoritative record defines bourbon requirements.",
                    "verified_fact",
                    0.95,
                    "Bourbon was formally described",
                )
            ]
        return [
            ExtractedClaim(
                "Bourbon was not created by a single inventor.",
                "supported_interpretation",
                0.8,
                "not created by a single inventor",
            )
        ]

    def find_contradictions(self, claims):
        return [(claims[0]["id"], claims[1]["id"])] if len(claims) >= 2 else []


def make_desk(tmp_path):
    return ResearchDesk(
        workspace=str(tmp_path / "desk"),
        search_provider=FakeSearch(),
        fetcher=FakeFetcher(),
        model=FakeModel(),
    )


def test_research_desk_runs_persistent_end_to_end_workflow(tmp_path):
    desk = make_desk(tmp_path)
    try:
        project = desk.create_project(
            "Documented origins of bourbon",
            "Separate documented history from repeated folklore.",
        )
        assert len(desk.plan(project.slug)) == 2

        dry_run = desk.run(project.slug, max_sources=2, dry_run=True)
        assert len(dry_run["discovered"]) == 2
        assert dry_run["stored_sources"] == 0

        result = desk.run(project.slug, max_sources=2)
        status = desk.status(project.slug)
        trace = desk.memory_trace(project.slug)

        assert result["stored_sources"] == 2
        assert result["stored_claims"] == 2
        assert status["sources"] == 2
        assert status["claims"] == 2
        assert status["open_contradictions"] == 1
        assert "authoritative record" in trace["context"]
        included = [
            decision for decision in trace["decisions"]
            if decision["reason"] == "included"
        ]
        assert included
        assert all(decision["component_id"].startswith("claim-") for decision in included)

        summary_id = desk.consolidate(project.slug)
        assert summary_id == "research-summary-1"
        report = desk.report(project.slug)
        report_text = report.read_text()
        assert "[S1]" in report_text
        assert "https://example.gov/history" in report_text
        assert desk.repository.session_changes(project.id)
    finally:
        desk.close()


def test_research_run_deduplicates_previously_read_urls(tmp_path):
    desk = make_desk(tmp_path)
    try:
        project = desk.create_project("Bourbon history")
        desk.plan(project.slug)
        desk.run(project.slug, max_sources=2)

        repeated = desk.run(project.slug, max_sources=2)

        assert repeated["stored_sources"] == 0
        assert repeated["stored_claims"] == 0
    finally:
        desk.close()


def test_source_classification_distinguishes_authority_and_folklore():
    assert classify_source("https://www.ttb.gov/spirits") == "primary_authority"
    assert classify_source("https://archive.example.edu/item") == "strong_secondary"
    assert classify_source("https://reddit.com/r/bourbon") == "community_or_folklore"


def test_cli_can_create_and_inspect_project(tmp_path, capsys):
    workspace = str(tmp_path / "cli")

    assert main(["--workspace", workspace, "project", "create", "Bourbon history"]) == 0
    assert main(["--workspace", workspace, "status"]) == 0

    output = capsys.readouterr().out
    assert "bourbon-history" in output
    assert "sources: 0" in output


def test_cli_loads_dotenv_before_constructing_desk(monkeypatch):
    calls = []

    class StubDesk:
        def __init__(self, workspace):
            calls.append(("desk", workspace))
            self.repository = self

        def list_projects(self):
            return []

        def close(self):
            calls.append(("close", None))

    def fake_load_dotenv(override):
        calls.append(("dotenv", override))

    monkeypatch.setattr(research_cli, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(research_cli, "ResearchDesk", StubDesk)

    assert research_cli.main(["project", "list"]) == 0
    assert calls == [
        ("dotenv", False),
        ("desk", ".bourbon-research"),
        ("close", None),
    ]


def test_openai_plan_accepts_structured_question_objects():
    model = object.__new__(OpenAIResearchModel)
    model._json = lambda system, user: {
        "questions": [
            {"id": "H-01", "question": "What is documented?", "priority": "high"},
            {"text": "Which stories are disputed?"},
            "How did the law change?",
        ]
    }

    assert model.plan("Bourbon", "Separate evidence from folklore") == [
        "What is documented?",
        "Which stories are disputed?",
        "How did the law change?",
    ]


def test_wikipedia_discovery_rejects_adjacent_topic_drift(tmp_path):
    class FakeWikipedia(WikipediaSearchProvider):
        def search(self, query, limit=10):
            return [
                SearchResult(
                    "https://en.wikipedia.org/wiki/Bourbon_whiskey",
                    "Bourbon whiskey",
                    "American whiskey made primarily from corn",
                ),
                SearchResult(
                    "https://en.wikipedia.org/wiki/Sake",
                    "Sake",
                    "Japanese alcoholic beverage",
                ),
                SearchResult(
                    "https://en.wikipedia.org/wiki/Japanese_whisky",
                    "Japanese whisky",
                    "Whisky produced in Japan",
                ),
            ][:limit]

    desk = ResearchDesk(
        workspace=str(tmp_path / "desk"),
        search_provider=FakeWikipedia(),
        fetcher=FakeFetcher(),
        model=FakeModel(),
    )
    try:
        project = desk.create_project("Bourbon whiskey")
        desk.plan(project.slug)
        results = desk.discover(project.slug, max_sources=5)
        assert [item["result"].title for item in results] == ["Bourbon whiskey"]
    finally:
        desk.close()


def test_openai_web_search_returns_deduplicated_sources():
    class Responses:
        def create(self, **kwargs):
            self.request = kwargs
            return {
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "sources": [
                                {
                                    "url": "https://www.ttb.gov/spirits",
                                    "title": "https://www.ttb.gov/spirits",
                                },
                                {
                                    "url": "https://www.loc.gov/item/123",
                                    "title": "Library of Congress record",
                                },
                            ]
                        },
                    },
                    {
                        "type": "message",
                        "content": [{
                            "annotations": [{
                                "type": "url_citation",
                                "url": "https://www.ttb.gov/spirits",
                                "title": "TTB distilled spirits",
                            }]
                        }],
                    },
                ]
            }

    class Client:
        def __init__(self):
            self.responses = Responses()

    client = Client()
    provider = OpenAIWebSearchProvider(client=client, model="search-test")
    results = provider.search("bourbon legal definition", limit=5)

    assert [result.url for result in results] == [
        "https://www.ttb.gov/spirits",
        "https://www.loc.gov/item/123",
    ]
    assert results[0].title == "TTB distilled spirits"
    assert client.responses.request["tools"][0]["type"] == "web_search"
    assert client.responses.request["include"] == ["web_search_call.action.sources"]


def test_openai_claim_extraction_normalizes_named_confidence():
    model = object.__new__(OpenAIResearchModel)
    model._json = lambda system, user: {
        "claims": [{
            "text": "Federal regulations define standards for bourbon.",
            "evidence_type": "verified_fact",
            "confidence": "high",
            "quotation": "standards of identity",
        }]
    }
    source = FetchedSource(
        "https://www.ttb.gov/example", "TTB", "Regulatory source text" * 20,
        "ttb.gov", "primary_authority",
    )

    claims = model.extract_claims(source)
    assert claims[0].confidence == 0.85
