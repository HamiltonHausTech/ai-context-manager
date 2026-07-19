import json

import pytest

from ai_context_manager.components import (
    ContextComponent,
    AgentGoalComponent,
    AgentSessionComponent,
    LongTermMemoryComponent,
    TaskSummaryComponent,
    UserProfileComponent,
)
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.store.json_memory import JSONMemoryStore
from ai_context_manager.store.errors import StorageReadError
from ai_context_manager.utils import (
    component_from_dict,
    component_to_dict,
    estimate_tokens,
    register_component_type,
)


@pytest.mark.parametrize(
    "component",
    [
        TaskSummaryComponent(
            "task-1", "Investigate retrieval", "Found a ranking issue", 2.5, ["task", "retrieval"]
        ),
        LongTermMemoryComponent(
            "memory-1", "Use explicit filters", "test", "2026-01-02T03:04:05", 0.75, ["memory"]
        ),
        UserProfileComponent(
            "user-1", "Ada", {"format": "concise", "citations": True}, 1.25, ["profile"]
        ),
        AgentGoalComponent(
            "goal-1",
            "Ship a stable memory layer",
            "agent-1",
            2.0,
            "paused",
            0.4,
            "2027-01-01T00:00:00",
            ["agent", "goal"],
            "2026-01-01T00:00:00",
            "2026-02-01T00:00:00",
        ),
        AgentSessionComponent(
            "session-1",
            "agent-1",
            "evaluation",
            "Compared retrieval results",
            42.5,
            False,
            ["agent", "session"],
            "2026-03-01T00:00:00",
        ),
    ],
)
def test_component_serialization_round_trip(component):
    serialized = component_to_dict(component)
    recovered = component_from_dict(component.id, serialized)

    assert serialized["schema_version"] == 1
    assert type(recovered) is type(component)
    assert component_to_dict(recovered) == serialized


def test_json_store_round_trip_is_lossless(tmp_path):
    path = tmp_path / "memory.json"
    original = AgentGoalComponent(
        "goal-1",
        "Preserve structured state",
        "agent-7",
        priority=2.25,
        status="paused",
        progress=0.6,
        deadline="2027-04-05T00:00:00",
        created_at="2026-01-01T00:00:00",
        last_updated="2026-02-01T00:00:00",
    )

    first = ContextManager(memory_store=JSONMemoryStore(str(path)))
    first.register_component(original)
    second = ContextManager(memory_store=JSONMemoryStore(str(path)))
    recovered = second.components[original.id]

    assert component_to_dict(recovered) == component_to_dict(original)
    raw = json.loads(path.read_text())
    assert raw[0]["schema_version"] == 1


def test_legacy_component_record_still_loads():
    recovered = component_from_dict(
        "old-task",
        {
            "id": "old-task",
            "type": "TaskSummaryComponent",
            "tags": ["task"],
            "content": "Task: old\nSummary: legacy text",
        },
    )

    assert recovered.task_name == "Recovered"
    assert recovered.summary == "Task: old\nSummary: legacy text"


def test_invalid_structured_payload_falls_back_to_legacy_content():
    recovered = component_from_dict(
        "task",
        {
            "type": "TaskSummaryComponent",
            "content": "legacy fallback",
            "component_data": "{not-json",
        },
    )

    assert recovered.task_name == "Recovered"
    assert recovered.summary == "legacy fallback"


def test_invalid_json_store_has_an_actionable_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not-json")

    with pytest.raises(StorageReadError, match="Cannot read JSON memory store"):
        JSONMemoryStore(str(path))


def test_context_order_is_stable_for_equal_scores():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("first", "First", "one"))
    manager.register_component(TaskSummaryComponent("second", "Second", "two"))

    metadata = manager.get_context(return_metadata=True)

    assert [item["id"] for item in metadata] == ["first", "second"]


def test_context_never_exceeds_token_budget():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("large", "Large", "word " * 200))

    context = manager.get_context(token_budget=10, summarize_if_needed=True)

    assert context
    assert estimate_tokens(context) <= 10


def test_tag_matching_preserves_current_any_semantics():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("one", "One", "one", tags=["alpha"]))
    manager.register_component(TaskSummaryComponent("two", "Two", "two", tags=["beta"]))

    metadata = manager.get_context(include_tags=["alpha", "beta"], return_metadata=True)

    assert {item["id"] for item in metadata} == {"one", "two"}


def test_tag_matching_can_require_all_tags():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("one", "One", "one", tags=["alpha", "beta"]))
    manager.register_component(TaskSummaryComponent("two", "Two", "two", tags=["alpha"]))

    metadata = manager.get_context(
        include_tags=["alpha", "beta"], tag_match_mode="all", return_metadata=True
    )

    assert [item["id"] for item in metadata] == ["one"]


def test_task_context_excludes_other_task_summaries():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("task-1", "One", "first"))
    manager.register_component(TaskSummaryComponent("task-2", "Two", "second"))
    manager.register_component(UserProfileComponent("profile", "Ada", {"style": "brief"}))

    metadata = manager.get_task_context_metadata("task-1")

    assert {item["id"] for item in metadata} == {"task-1", "profile"}


def test_invalid_tag_match_mode_is_rejected():
    manager = ContextManager()

    with pytest.raises(ValueError, match="tag_match_mode"):
        manager.get_context(tag_match_mode="sometimes")


def test_duplicate_registration_policy_is_explicit():
    manager = ContextManager()
    first = TaskSummaryComponent("task", "First", "first")
    replacement = TaskSummaryComponent("task", "Replacement", "replacement")
    manager.register_component(first)

    manager.register_component(replacement)
    assert manager.components["task"] is first

    manager.register_component(replacement, on_duplicate="replace")
    assert manager.components["task"] is replacement

    with pytest.raises(ValueError, match="already registered"):
        manager.register_component(first, on_duplicate="error")


def test_dry_run_returns_structured_trace(capsys):
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("task", "Trace", "inspect selection"))

    trace = manager.get_context(dry_run=True)

    assert trace[0]["id"] == "task"
    assert trace[0]["tokens"] > 0
    assert trace[0]["summarized"] is False
    assert "Dry Run Complete" in capsys.readouterr().out


def test_custom_component_can_register_round_trip_codec():
    class CustomComponent(ContextComponent):
        def __init__(self, id, value, tags=None):
            super().__init__(id, tags)
            self.value = value

        def load_content(self):
            return f"Custom: {self.value}"

    register_component_type(
        "CustomComponent",
        lambda component: {"value": component.value},
        lambda component_id, fields, tags: CustomComponent(
            component_id, fields["value"], tags
        ),
    )
    original = CustomComponent("custom-1", {"nested": [1, 2]}, ["custom"])

    recovered = component_from_dict(original.id, component_to_dict(original))

    assert isinstance(recovered, CustomComponent)
    assert recovered.value == original.value
    assert recovered.tags == original.tags


def test_token_estimation_falls_back_when_encoding_cannot_load(monkeypatch):
    from ai_context_manager import tokenization

    class UnavailableTiktoken:
        @staticmethod
        def get_encoding(_name):
            raise OSError("offline")

    monkeypatch.setattr(tokenization, "_encoding", None)
    monkeypatch.setattr(tokenization, "_encoding_loaded", False)
    monkeypatch.setattr(tokenization, "_fallback_warning_emitted", False)
    monkeypatch.setattr(tokenization, "import_module", lambda _name: UnavailableTiktoken)

    assert tokenization.estimate_tokens("one two three") == 3
