from abc import ABC, abstractmethod
from io import BytesIO
from html.parser import HTMLParser
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests

from bourbon_research.models import ExtractedClaim, FetchedSource, SearchResult


USER_AGENT = "BourbonResearchDesk/0.1 (+local research application)"


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        pass


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._href = None
        self._title = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a" and "result-link" in attributes.get("class", ""):
            self._href = attributes.get("href")
            self._title = []

    def handle_data(self, data):
        if self._href:
            self._title.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            href = self._href
            parsed = urlparse(href)
            if "duckduckgo.com" in parsed.netloc and parse_qs(parsed.query).get("uddg"):
                href = unquote(parse_qs(parsed.query)["uddg"][0])
            if href.startswith("http"):
                self.results.append(SearchResult(href, " ".join(self._title).strip()))
            self._href = None


class DuckDuckGoSearchProvider(SearchProvider):
    """No-key development search provider using DuckDuckGo's HTML endpoint."""

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        if "anomaly-modal" in response.text or "Unfortunately, bots use DuckDuckGo" in response.text:
            raise RuntimeError(
                "DuckDuckGo challenged the automated request; configure Brave search "
                "or use the Wikipedia development provider"
            )
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        return parser.results[:limit]


class WikipediaSearchProvider(SearchProvider):
    """Stable no-key discovery provider for development and pipeline testing."""

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(limit, 20),
                "format": "json",
                "utf8": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        results = []
        for item in response.json().get("query", {}).get("search", []):
            title = item["title"]
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            results.append(
                SearchResult(
                    f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    title,
                    snippet,
                )
            )
        return results


class BraveSearchProvider(SearchProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY is required for Brave search")

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(limit, 20)},
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            timeout=20,
        )
        response.raise_for_status()
        return [
            SearchResult(item["url"], item.get("title", item["url"]), item.get("description", ""))
            for item in response.json().get("web", {}).get("results", [])
        ]


class OpenAIWebSearchProvider(SearchProvider):
    """Web discovery through the Responses API using the existing OpenAI key."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client=None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.client = client
        self.model = model or os.getenv(
            "RESEARCH_SEARCH_MODEL", os.getenv("RESEARCH_MODEL", "gpt-5-mini")
        )

    @staticmethod
    def _value(item, name, default=None):
        return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        response = self.client.responses.create(
            model=self.model,
            tools=[{
                "type": "web_search",
                "search_context_size": "low",
                "filters": {
                    "blocked_domains": [
                        "wikipedia.org", "reddit.com", "quora.com", "scribd.com",
                        "pinterest.com", "facebook.com", "x.com",
                    ]
                },
            }],
            tool_choice="required",
            include=["web_search_call.action.sources"],
            input=(
                f"Search the web for: {query}\n"
                f"Identify up to {limit} directly relevant source pages. Prioritize "
                "government records, statutes, archives, libraries, museums, academic "
                "work, and carefully sourced specialist history. Avoid retail pages, "
                "forums, generic listicles, and unsupported summaries. Cite every source."
            ),
        )
        results = []
        seen = set()

        def add(url, title=None, snippet=""):
            if not url or not url.startswith("http"):
                return
            if url in seen:
                for index, existing in enumerate(results):
                    if existing.url == url and existing.title == url and title:
                        results[index] = SearchResult(url, title, existing.snippet or snippet)
                return
            seen.add(url)
            results.append(SearchResult(url, title or url, snippet))

        for output in self._value(response, "output", []) or []:
            if self._value(output, "type") == "web_search_call":
                action = self._value(output, "action", {})
                for source in self._value(action, "sources", []) or []:
                    add(
                        self._value(source, "url"),
                        self._value(source, "title"),
                        self._value(source, "snippet", ""),
                    )
            if self._value(output, "type") == "message":
                for content in self._value(output, "content", []) or []:
                    for annotation in self._value(content, "annotations", []) or []:
                        if self._value(annotation, "type") == "url_citation":
                            add(
                                self._value(annotation, "url"),
                                self._value(annotation, "title"),
                            )
        return results[:limit]


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.title = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title.append(data)
        self.parts.append(data)


def classify_source(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.endswith(".gov") or ".gov." in host:
        return "primary_authority"
    if host.endswith(".edu") or any(term in host for term in ("archive", "museum", "loc.gov")):
        return "strong_secondary"
    if any(term in host for term in ("wikipedia", "britannica")):
        return "reference"
    if any(term in host for term in ("reddit", "forum", "blogspot")):
        return "community_or_folklore"
    if any(term in host for term in ("totalwine", "reservebar", "drizly")):
        return "retail"
    return "web_secondary"


class SourceFetcher:
    def _fetch_wikipedia(self, result: SearchResult) -> Optional[FetchedSource]:
        parsed = urlparse(result.url)
        if parsed.netloc.lower() != "en.wikipedia.org" or not parsed.path.startswith("/wiki/"):
            return None
        title = unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        text = page.get("extract", "").strip()
        if len(text) < 200:
            raise ValueError("Wikipedia source contains too little extractable text")
        return FetchedSource(
            result.url,
            page.get("title", result.title),
            text,
            parsed.netloc,
            classify_source(result.url),
        )

    def fetch(self, result: SearchResult, max_bytes: int = 2_000_000) -> FetchedSource:
        wikipedia_source = self._fetch_wikipedia(result)
        if wikipedia_source is not None:
            return wikipedia_source
        response = requests.get(
            result.url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            stream=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if not any(kind in content_type for kind in ("text/html", "text/plain", "application/pdf")):
            raise ValueError(f"Unsupported content type: {content_type or 'unknown'}")
        content = response.content[:max_bytes]
        if "application/pdf" in content_type:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
            title = str(reader.metadata.title or result.title) if reader.metadata else result.title
        else:
            decoded = content.decode(response.encoding or "utf-8", errors="replace")
        if "text/html" in content_type:
            parser = _TextExtractor()
            parser.feed(decoded)
            text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            title = " ".join(parser.title).strip() or result.title
        elif "text/plain" in content_type:
            text, title = decoded.strip(), result.title
        if len(text) < 200:
            raise ValueError("Source contains too little extractable text")
        return FetchedSource(
            result.url,
            title,
            text,
            urlparse(result.url).netloc,
            classify_source(result.url),
        )


class ResearchModel(ABC):
    @abstractmethod
    def plan(self, subject: str, objective: str) -> List[str]:
        pass

    @abstractmethod
    def extract_claims(self, source: FetchedSource) -> List[ExtractedClaim]:
        pass

    @abstractmethod
    def find_contradictions(self, claims: List[Dict]) -> List[Tuple[int, int]]:
        pass


class HeuristicResearchModel(ResearchModel):
    """Offline fallback useful for pipeline tests; not a substitute for review."""

    def plan(self, subject: str, objective: str) -> List[str]:
        return [
            f"What is the earliest reliable documentary evidence for the history and origin of {subject}?",
            f"How have legal definitions and regulations governing {subject} changed?",
            f"Which commonly repeated origin stories about {subject} are disputed or weakly sourced?",
            f"Which production methods and terminology are essential to understanding {subject}?",
            f"How does {subject} compare with Scotch whisky and Irish whiskey?",
        ]

    def extract_claims(self, source: FetchedSource) -> List[ExtractedClaim]:
        sentences = re.split(r"(?<=[.!?])\s+", source.text)
        evidence = {
            "primary_authority": "verified_fact",
            "strong_secondary": "supported_interpretation",
            "community_or_folklore": "unverified_story",
        }.get(source.source_class, "source_claim")
        claims = []
        for sentence in sentences:
            cleaned = " ".join(sentence.split())
            if (
                60 <= len(cleaned) <= 500
                and cleaned[0].isupper()
                and re.search(r"\b(is|was|were|became|requires?|began|first)\b", cleaned, re.I)
            ):
                claims.append(ExtractedClaim(cleaned, evidence, 0.55, cleaned))
            if len(claims) >= 12:
                break
        return claims

    def find_contradictions(self, claims: List[Dict]) -> List[Tuple[int, int]]:
        pairs = []
        for index, left in enumerate(claims):
            left_words = set(re.findall(r"[a-z]{4,}", left["text"].lower()))
            for right in claims[index + 1:]:
                right_words = set(re.findall(r"[a-z]{4,}", right["text"].lower()))
                overlap = len(left_words & right_words) / max(1, len(left_words | right_words))
                negation_diff = (" not " in f" {left['text'].lower()} ") != (" not " in f" {right['text'].lower()} ")
                if overlap >= 0.5 and negation_diff:
                    pairs.append((left["id"], right["id"]))
        return pairs


class OpenAIResearchModel(ResearchModel):
    def __init__(self, model: str = "gpt-5-mini", api_key: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _json(self, system: str, user: str) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def plan(self, subject: str, objective: str) -> List[str]:
        data = self._json(
            "Create a rigorous research plan. Return JSON with a questions array. Separate documented history, law, production, comparisons, and disputed stories.",
            f"Subject: {subject}\nObjective: {objective}",
        )
        questions = []
        for item in data.get("questions", []):
            if isinstance(item, str):
                question = item
            elif isinstance(item, dict):
                question = item.get("question") or item.get("text")
            else:
                question = None
            if question and str(question).strip():
                questions.append(str(question).strip())
        if not questions:
            raise ValueError("Research model returned no usable questions")
        return questions

    def extract_claims(self, source: FetchedSource) -> List[ExtractedClaim]:
        data = self._json(
            "Extract material claims from a research source. Return JSON {claims:[{text,evidence_type,confidence,quotation}]}. evidence_type must be verified_fact, supported_interpretation, producer_claim, opinion, source_claim, or unverified_story. Do not invent quotations.",
            f"URL: {source.url}\nClass: {source.source_class}\nText:\n{source.text[:50000]}",
        )
        confidence_names = {"low": 0.35, "medium": 0.65, "high": 0.85}

        def confidence(value) -> float:
            if isinstance(value, str) and value.lower() in confidence_names:
                return confidence_names[value.lower()]
            return max(0.0, min(1.0, float(value)))

        return [
            ExtractedClaim(
                item["text"], item.get("evidence_type", "source_claim"),
                confidence(item.get("confidence", 0.5)),
                item.get("quotation"),
            )
            for item in data.get("claims", [])
        ]

    def find_contradictions(self, claims: List[Dict]) -> List[Tuple[int, int]]:
        compact = [{"id": item["id"], "text": item["text"]} for item in claims]
        data = self._json(
            "Identify genuinely incompatible claims. Return JSON {pairs:[{left_id,right_id}]}. Do not mark mere differences in scope or opinion as contradictions.",
            json.dumps(compact)[:60000],
        )
        return [(int(item["left_id"]), int(item["right_id"])) for item in data.get("pairs", [])]


def default_search_provider() -> SearchProvider:
    configured = os.getenv("RESEARCH_SEARCH_PROVIDER", "").lower()
    if configured == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for OpenAI web search")
        return OpenAIWebSearchProvider()
    if configured == "brave":
        return BraveSearchProvider()
    if configured == "duckduckgo":
        return DuckDuckGoSearchProvider()
    if configured == "wikipedia":
        return WikipediaSearchProvider()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIWebSearchProvider()
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        return BraveSearchProvider()
    return WikipediaSearchProvider()


def default_research_model() -> ResearchModel:
    return OpenAIResearchModel(model=os.getenv("RESEARCH_MODEL", "gpt-5-mini")) if os.getenv("OPENAI_API_KEY") else HeuristicResearchModel()
