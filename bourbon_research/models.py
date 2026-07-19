from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Project:
    id: int
    slug: str
    subject: str
    objective: str
    created_at: str


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str = ""


@dataclass(frozen=True)
class FetchedSource:
    url: str
    title: str
    text: str
    publisher: str
    source_class: str
    published_at: Optional[str] = None


@dataclass(frozen=True)
class ExtractedClaim:
    text: str
    evidence_type: str
    confidence: float
    quotation: Optional[str] = None
