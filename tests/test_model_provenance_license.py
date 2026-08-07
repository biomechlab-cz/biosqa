"""The shipped weights carry restricted-data lineage — the public tree must say so.

The four `.onnx` files are git-tracked in a repo with a public remote and a Pages
deploy, under a bare MIT LICENSE that covers "the Software". Before this suite
existed, `grep -i weights` over LICENSE, README.md, models/README.md and every
docs page returned zero hits: nothing told a downstream user that eeg.onnx has
TUAR lineage or that ppg.onnx/eda.onnx trace back to credentialed and
non-commercial cohorts. These tests pin the carve-out (and the machine-readable
`license` block) so it cannot silently disappear again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biosqa.model.model_card import load_model_card

APP = Path(__file__).resolve().parents[1]
MODELS = APP / "models"
CARDS = sorted(MODELS.glob("*.model_card.json"))

REDISTRIBUTION_VALUES = {"attribution-required", "review-required", "restricted"}

pytestmark = pytest.mark.skipif(not CARDS, reason="no shipped model cards in app/models")


def _raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


#: Datasets whose terms an openly-licensed weight release cannot honour: PhysioNet
#: credentialed access (MIMIC), non-commercial only (WESAD), or a signed data use
#: agreement that is silent on derived works (TUAR). No SHIPPED model may have any of
#: these in its lineage -- that is the whole reason eeg.onnx and ppg.onnx are not here.
RESTRICTED_SOURCES = ("TUAR", "MIMIC", "WESAD")


def test_license_models_exists_and_covers_every_shipped_model():
    text = (APP / "LICENSE-MODELS").read_text(encoding="utf-8")
    assert "MIT" in text and "LICENSE" in text
    for path in CARDS:
        assert f"{path.name.split('.')[0]}.onnx" in text
    # and it explains why the other two modalities have no weights, naming the sources
    # responsible -- a reader must not have to guess whether they were forgotten
    for source in RESTRICTED_SOURCES:
        assert source in text


def test_no_shipped_model_has_restricted_lineage():
    """The compliance guard. Removing eeg/ppg is only durable if re-adding a model with
    credentialed, non-commercial or DUA-bound training data fails loudly."""
    for path in CARDS:
        terms = " ".join(_raw(path)["license"]["source_terms"])
        cohorts = " ".join(_raw(path)["training_data_provenance"]["cohorts"])
        for source in RESTRICTED_SOURCES:
            assert source not in terms and source not in cohorts, (
                f"{path.name} declares {source} in its lineage, but it is shipped in a "
                f"repo whose weights are offered under attribution-only terms")
        assert _raw(path)["license"]["redistribution"] == "attribution-required"


def test_the_withheld_weights_are_actually_absent():
    for modality in ("eeg", "ppg"):
        assert not (MODELS / f"{modality}.onnx").exists(), (
            f"{modality}.onnx is back in models/ -- see LICENSE-MODELS for why it is withheld")
        assert not (MODELS / f"{modality}.model_card.json").exists()


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.stem)
def test_card_declares_its_licence_machine_readably(path):
    lic = _raw(path).get("license")
    assert lic is not None, f"{path.name} has no `license` block"
    assert lic["redistribution"] in REDISTRIBUTION_VALUES
    assert "MIT" in lic["code_license"]
    assert (APP / lic["weights_terms"].replace("app/", "")).is_file()
    assert lic["source_terms"]


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.stem)
def test_card_provenance_is_a_digest_or_says_why_not(path):
    raw = _raw(path)
    assert raw["training_data_hash"], "empty provenance passes the loader's presence-only check"
    prov = raw.get("training_data_provenance")
    assert prov is not None, f"{path.name} has no `training_data_provenance` block"
    if prov.get("digest"):
        # a real digest is a sha256:-prefixed 64-char hex
        digest = prov["digest"].removeprefix("sha256:")
        assert len(digest) == 64 and not digest.strip("0123456789abcdef")
        assert prov["digest_source"]
    else:
        assert prov["digest_note"], "no digest and no explanation of why"


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.stem)
def test_licence_and_provenance_blocks_do_not_break_the_loader(path):
    # Unknown top-level keys must stay inert: the card contract is validated by
    # field, not by whitelist, and these two blocks are documentation only.
    card = load_model_card(path)
    assert card.modality == path.name.split(".")[0]


def test_public_docs_carry_the_weights_carve_out():
    # This is the exact grep that returned zero hits before the fix.
    for rel in ("README.md", "models/README.md", "docs/src/content/docs/contributing.md"):
        text = (APP / rel).read_text(encoding="utf-8").lower()
        assert "weights" in text, f"{rel} says nothing about the model weights"
        assert "license-models" in text or "licence-models" in text


def test_readme_dataset_attribution_matches_what_was_ingested():
    text = (APP / "README.md").read_text(encoding="utf-8")
    # credited but never ingested: no ltstdb/macecgdb loader output is in any store
    assert "Long-Term ST" not in text
    assert "macecgdb" not in text
    # used but previously uncredited in the per-modality lists
    assert text.count("PPG-DaLiA") >= 2  # PPG list + EDA list
