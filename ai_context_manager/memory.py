"""Memory taxonomy and lifecycle state shared by all context components."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryKind(str, Enum):
    GENERIC = "generic"
    EPISODE = "episode"
    DURABLE_FACT = "durable_fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    DERIVED_SUMMARY = "derived_summary"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


@dataclass
class MemoryLifecycle:
    kind: str = MemoryKind.GENERIC.value
    status: str = MemoryStatus.ACTIVE.value
    provenance_ids: List[str] = field(default_factory=list)
    supersedes_ids: List[str] = field(default_factory=list)
    contradiction_ids: List[str] = field(default_factory=list)
    expires_at: Optional[str] = None
    confidence: float = 1.0

    def validate(self) -> None:
        valid_kinds = {kind.value for kind in MemoryKind}
        valid_statuses = {status.value for status in MemoryStatus}
        if self.kind not in valid_kinds:
            raise ValueError(f"Unknown memory kind: {self.kind}")
        if self.status not in valid_statuses:
            raise ValueError(f"Unknown memory status: {self.status}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Memory confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MemoryLifecycle":
        lifecycle = cls(**(data or {}))
        lifecycle.validate()
        return lifecycle

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.status == MemoryStatus.EXPIRED.value:
            return True
        if not self.expires_at:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            return current >= expires
        except ValueError:
            return False

    def is_active(self, now: Optional[datetime] = None) -> bool:
        return self.status == MemoryStatus.ACTIVE.value and not self.is_expired(now)


def memory_record_is_active(record: Dict[str, Any]) -> bool:
    """Treat legacy records as active and apply lifecycle state when available."""
    memory = record.get("memory")
    if memory is None and isinstance(record.get("metadata"), dict):
        memory = record["metadata"].get("memory")
    return MemoryLifecycle.from_dict(memory).is_active() if memory else True
