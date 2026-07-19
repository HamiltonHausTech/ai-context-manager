import json

import pytest

from ai_context_manager.components import TaskSummaryComponent
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.store.errors import StorageReadError, StorageWriteError
from ai_context_manager.store.json_memory import JSONMemoryStore
from ai_context_manager.store.base import MemoryStore
from ai_context_manager.store.sqlite_memory import SQLiteMemoryStore
from ai_context_manager.utils import component_to_dict
from ai_context_manager.utils import load_stores_from_config


@pytest.fixture(params=["json", "sqlite"])
def memory_store(request, tmp_path):
    if request.param == "json":
        store = JSONMemoryStore(str(tmp_path / "memory.json"))
    else:
        store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    yield store
    if hasattr(store, "close"):
        store.close()


def test_memory_store_contract_create_read_update_delete(memory_store):
    original = component_to_dict(TaskSummaryComponent("task", "Original", "first"))
    updated = component_to_dict(TaskSummaryComponent("task", "Updated", "second"))

    memory_store.save_component(original)
    assert memory_store.get_component("task") == original
    assert memory_store.load_all() == [original]

    memory_store.save_component(updated)
    assert memory_store.get_component("task") == updated
    assert memory_store.load_all() == [updated]

    memory_store.delete_component("task")
    assert memory_store.get_component("task") is None
    assert memory_store.load_all() == []


def test_memory_store_contract_rejects_empty_id(memory_store):
    with pytest.raises(ValueError, match="non-empty ID"):
        memory_store.save_component({"id": ""})


def test_context_manager_round_trips_through_memory_store(memory_store):
    original = TaskSummaryComponent("task", "Persisted", "structured", score=2.0)
    ContextManager(memory_store=memory_store).register_component(original)

    recovered = ContextManager(memory_store=memory_store).components["task"]

    assert component_to_dict(recovered) == component_to_dict(original)


def test_json_write_is_atomic_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    store = JSONMemoryStore(str(path))
    original = {"id": "original", "value": 1}
    store.save_component(original)

    def fail_replace(_source, _destination):
        raise OSError("disk unavailable")

    monkeypatch.setattr("ai_context_manager.store.json_memory.os.replace", fail_replace)

    with pytest.raises(StorageWriteError, match="Cannot write JSON memory store"):
        store.save_component({"id": "new", "value": 2})

    assert json.loads(path.read_text()) == [original]
    assert store.load_all() == [original]


def test_invalid_json_raises_typed_read_error(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{broken")

    with pytest.raises(StorageReadError, match="Cannot read JSON memory store"):
        JSONMemoryStore(str(path))


def test_context_manager_does_not_hide_storage_write_failure():
    class FailingStore(MemoryStore):
        def load_all(self):
            return []

        def save_component(self, _component):
            raise StorageWriteError("write failed")

        def delete_component(self, _component_id):
            raise StorageWriteError("delete failed")

        def get_component(self, _component_id):
            return None

    manager = ContextManager(memory_store=FailingStore())

    with pytest.raises(StorageWriteError, match="write failed"):
        manager.register_component(TaskSummaryComponent("task", "Task", "result"))

    assert "task" not in manager.components


def test_sqlite_memory_store_loads_from_config(tmp_path):
    _feedback, memory = load_stores_from_config(
        {
            "feedback_store": {"type": "json", "filepath": str(tmp_path / "feedback.json")},
            "memory_store": {"type": "sqlite", "db_path": str(tmp_path / "memory.db")},
        }
    )
    try:
        assert isinstance(memory, SQLiteMemoryStore)
    finally:
        memory.close()
