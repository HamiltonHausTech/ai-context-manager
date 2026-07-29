from pathlib import Path

import experiments.adaptive_selection as adaptive_selection
import toml
from setuptools import find_packages

EXPECTED_PUBLIC_NAMES = {
    "DATASET_BUNDLE_VERSION",
    "TINY_CLAIM_LIMIT",
    "TINY_ONTOLOGY_CONFIG",
    "TINY_ONTOLOGY_CONFIG_HASH",
    "TINY_ONTOLOGY_DEFINITIONS",
    "TINY_ONTOLOGY_ID",
    "TINY_ONTOLOGY_VERSION",
    "DatasetBundle",
    "FamilyPlan",
    "canonical_bundle_sha256",
    "count_context_tokens",
    "load_dataset_bundle",
    "load_tiny_fixture",
    "validate_tiny_fixture",
    "ExperimentRepository",
    "EvidenceEntry",
    "IntegrityReport",
    "RepositoryError",
    "RepositoryDatabaseError",
    "DuplicateRecordError",
    "ReferenceIntegrityError",
    "RecordNotFoundError",
    "IntegrityError",
    "AdaptivePolicySelector",
    "FullContextSelector",
    "SelectionResult",
    "SelectorDecision",
    "SimilarityTopKSelector",
    "StaticPolicySelector",
    "SCHEMA_VERSION",
    "FEEDBACK_SIGNAL_TYPES",
    "FEEDBACK_SOURCES",
    "TaskProfile",
    "ContextItem",
    "CriterionScore",
    "RubricCriterion",
    "ScoringRubric",
    "TaskInputs",
    "SealedEvaluation",
    "TaskCase",
    "RunManifest",
    "SelectionDecision",
    "TaskOutcome",
    "FeedbackEvent",
    "UtilityEstimate",
    "ExperimentResult",
}
EXPECTED_FIXTURE_HASH = (
    "05cdb3edbc96d753f44b7161dcae8812679d7776306ebb6887911dda2f7bca32"
)


def test_setuptools_discovery_packages_adaptive_experiments():
    root = Path(__file__).parents[2]
    configuration = toml.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    includes = configuration["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "experiments*" in includes
    discovered = find_packages(where=root, include=includes)
    assert "experiments" in discovered
    assert "experiments.adaptive_selection" in discovered


def test_public_package_exports_load_and_persist_tiny_fixture(tmp_path):
    assert len(adaptive_selection.__all__) == len(set(adaptive_selection.__all__))
    assert set(adaptive_selection.__all__) == EXPECTED_PUBLIC_NAMES

    fixture = Path(__file__).parent / "fixtures" / "tiny_experiment.json"
    bundle = adaptive_selection.load_tiny_fixture(fixture)
    assert isinstance(bundle, adaptive_selection.DatasetBundle)
    assert len(bundle.cases) == 6
    assert adaptive_selection.canonical_bundle_sha256(bundle) == EXPECTED_FIXTURE_HASH

    feedback_events = list(bundle.adaptation_feedback)
    assert len(feedback_events) == 4
    assert all(
        isinstance(event, adaptive_selection.FeedbackEvent) for event in feedback_events
    )

    with adaptive_selection.ExperimentRepository(
        tmp_path / "public-api.sqlite3", clock=lambda: "2026-07-28T13:00:00Z"
    ) as repository:
        for run_id in dict.fromkeys(event.run_id for event in feedback_events):
            repository.append_run(
                adaptive_selection.RunManifest(
                    run_id=run_id,
                    experiment_version="public-api-integration-v1",
                    protocol_version="stage0-v2",
                    dataset_version=bundle.dataset_version,
                    dataset_hash=EXPECTED_FIXTURE_HASH,
                    selector_mode="adaptive",
                    selector_version="public-api-test-v1",
                    provider="test-provider",
                    model_id="test-model",
                    prompt_template_hash="sha256:test-prompt",
                    config_hash="sha256:test-config",
                    code_revision="test-revision",
                    temperature=0.0,
                    seed=0,
                    seed_supported=True,
                    tool_availability=(),
                    started_timestamp="2026-07-28T12:00:00Z",
                    provenance="test:public-package-integration",
                )
            )
        for event in feedback_events:
            repository.append_feedback(event)

        report = repository.verify_integrity()
        assert isinstance(report, adaptive_selection.IntegrityReport)
        assert report.ok
        assert report.evidence_rows == 8
        assert report.per_type_counts == {
            "feedback_event": 4,
            "run_manifest": 4,
        }
        assert repository.list_feedback() == feedback_events
