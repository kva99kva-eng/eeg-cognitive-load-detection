from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_key_sections():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    expected_sections = [
        "## Executive Summary",
        "## Project Goal",
        "## Why This Project Matters",
        "## Methodology",
        "## Window-Level vs Subject-Independent Validation",
        "## Limitations",
    ]

    for section in expected_sections:
        assert section in readme, f"Missing README section: {section}"


def test_readme_mentions_leakage_and_subject_independent_validation():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "leakage" in readme
    assert "subject-independent" in readme
    assert "window-level" in readme


def test_readme_has_no_common_encoding_artifacts():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    bad_fragments = ["вЂ", "В±", "в”", "В·"]

    for fragment in bad_fragments:
        assert fragment not in readme, f"Encoding artifact found in README: {fragment}"
