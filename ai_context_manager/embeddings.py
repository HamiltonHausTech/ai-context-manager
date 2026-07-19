"""Provider-independent embedding contracts and version metadata."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from typing import Any, Dict, List, Optional


class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""


class EmbeddingUnavailableError(EmbeddingError):
    """The configured provider cannot currently generate embeddings."""


class EmbeddingDimensionError(EmbeddingError):
    """A provider returned a vector with an incompatible dimension."""


@dataclass(frozen=True)
class EmbeddingInfo:
    provider: str
    model: str
    version: str
    dimension: int

    @property
    def identifier(self) -> str:
        return f"{self.provider}:{self.model}:{self.version}:{self.dimension}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def info(self) -> EmbeddingInfo:
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

    def embed_checked(self, text: str) -> List[float]:
        vector = [float(value) for value in self.embed(text)]
        if len(vector) != self.info.dimension:
            raise EmbeddingDimensionError(
                f"Provider {self.info.identifier} returned {len(vector)} values; "
                f"expected {self.info.dimension}"
            )
        return vector


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Lazy local sentence-transformers provider.

    Model loading is deferred until the first embedding request, so importing and
    configuring the package remains offline-safe.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        revision: Optional[str] = None,
    ):
        self.model_name = model
        self.dimension = dimension
        self.revision = revision
        self._model = None

    @property
    def info(self) -> EmbeddingInfo:
        try:
            package_version = importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError:
            package_version = "unavailable"
        revision = self.revision or "default"
        return EmbeddingInfo(
            "sentence-transformers",
            self.model_name,
            f"{package_version}@{revision}",
            self.dimension,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"revision": self.revision} if self.revision else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
            return self._model
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Cannot load embedding model {self.model_name}: {exc}"
            ) from exc

    def embed(self, text: str) -> List[float]:
        try:
            vector = self._load_model().encode(text)
            return vector.tolist() if hasattr(vector, "tolist") else list(vector)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Cannot generate embedding with {self.model_name}: {exc}"
            ) from exc


def stable_content_hash(text: str) -> str:
    """Stable identifier used to decide whether a record needs re-embedding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_embedding_text(component: Dict[str, Any]) -> str:
    """Build identical searchable text regardless of the vector backend."""
    content = component.get("content", "")
    component_type = component.get("type", "Unknown")
    tags = component.get("tags", [])
    lines = [content, f"Type: {component_type}"]
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    if component_type == "AgentGoalComponent":
        lines.append(f"Goal: {content}")
    elif component_type == "TaskSummaryComponent":
        lines.append(f"Task: {component.get('task_name', 'Unknown')}")
    elif component_type == "LongTermMemoryComponent":
        lines.append(f"Learning: {content}")
    return "\n".join(lines) + "\n"


def embedding_metadata(provider: EmbeddingProvider, text: str) -> Dict[str, Any]:
    info = provider.info
    return {
        "embedding_provider": info.provider,
        "embedding_model": info.model,
        "embedding_version": info.version,
        "embedding_dimension": info.dimension,
        "embedding_identifier": info.identifier,
        "embedding_content_hash": stable_content_hash(text),
    }


def needs_reembedding(
    metadata: Dict[str, Any], provider: EmbeddingProvider, text: str
) -> bool:
    expected = embedding_metadata(provider, text)
    return any(metadata.get(key) != value for key, value in expected.items())
