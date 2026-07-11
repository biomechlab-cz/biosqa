"""Unit tests for the Plan 1 <-> Plan 2 model_card.json handshake (Plan 2 §11)."""

from __future__ import annotations

import json

import pytest

from biosqa.model.model_card import (
    ModelCardError,
    load_model_card,
    validate_onnx_input_shape,
)

VALID_CARD = {
    "modality": "ecg",
    "L_m": 2048,
    "fs_hz": 250,
    "class_order": ["Q0", "Q1", "Q2", "Q3"],
    "normalization": {"method": "zscore", "mean": 0.0, "std": 1.0},
    "training_data_hash": "sha256:deadbeef",
    "model_version": "0.1.0",
}


def _write_card(tmp_path, data) -> str:
    path = tmp_path / "ecg.model_card.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_load_model_card_valid(tmp_path):
    card = load_model_card(_write_card(tmp_path, VALID_CARD))
    assert card.modality == "ecg"
    assert card.l_m == 2048
    assert card.class_order == ("Q0", "Q1", "Q2", "Q3")
    assert card.n_classes == 4
    assert card.normalization.method == "zscore"


def test_load_model_card_missing_file(tmp_path):
    with pytest.raises(ModelCardError):
        load_model_card(tmp_path / "does_not_exist.json")


def test_load_model_card_missing_required_field(tmp_path):
    bad = dict(VALID_CARD)
    del bad["fs_hz"]
    with pytest.raises(ModelCardError):
        load_model_card(_write_card(tmp_path, bad))


def test_load_model_card_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(ModelCardError):
        load_model_card(str(path))


def test_load_model_card_empty_class_order(tmp_path):
    bad = dict(VALID_CARD)
    bad["class_order"] = []
    with pytest.raises(ModelCardError):
        load_model_card(_write_card(tmp_path, bad))


def test_validate_onnx_input_shape_matches():
    card = load_model_card_from_dict(VALID_CARD)
    validate_onnx_input_shape(card, (1, 1, 2048))  # should not raise


def test_validate_onnx_input_shape_mismatch():
    card = load_model_card_from_dict(VALID_CARD)
    with pytest.raises(ModelCardError):
        validate_onnx_input_shape(card, (1, 1, 999))


def load_model_card_from_dict(data, tmp_path_factory=None):
    """Helper: round-trip through a temp file since `load_model_card` reads from disk."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "card.json"
        path.write_text(json.dumps(data))
        return load_model_card(path)


# --- v2 multi-head contract (Plan 1 §12.1) ----------------------------------

V2_CARD = {
    **VALID_CARD,
    "class_order": ["Q0", "Q1", "Q2", "Q3"],
    "heads": [
        {"name": "grade", "output_name": "q_logits", "kind": "ordinal",
         "activation": "softmax", "class_order": ["Q0", "Q1", "Q2", "Q3"]},
        {"name": "usable", "output_name": "bin_logits", "kind": "binary",
         "activation": "softmax", "class_order": ["BAD", "OK"]},
        {"name": "artifact", "output_name": "type_logits", "kind": "multilabel",
         "activation": "sigmoid",
         "class_order": ["clean", "baseline_wander", "motion", "muscle", "electrode", "powerline"],
         "threshold": 0.5},
    ],
}


def test_legacy_card_synthesizes_single_grade_head():
    card = load_model_card_from_dict(VALID_CARD)
    assert len(card.heads) == 1
    grade = card.heads[0]
    assert (grade.name, grade.output_name, grade.kind, grade.activation) == (
        "grade", "q_logits", "ordinal", "softmax",
    )
    assert grade.class_order == card.class_order == ("Q0", "Q1", "Q2", "Q3")
    assert card.primary_head is grade
    assert card.artifact_head is None
    assert card.n_classes == 4


def test_v2_heads_parsed():
    card = load_model_card_from_dict(V2_CARD)
    assert [h.name for h in card.heads] == ["grade", "usable", "artifact"]
    assert card.primary_head.name == "grade"
    assert card.n_classes == 4  # still driven by the ordinal head
    usable = card.head("usable")
    assert usable.kind == "binary" and usable.class_order == ("BAD", "OK")
    artifact = card.artifact_head
    assert artifact is not None
    assert artifact.kind == "multilabel" and artifact.activation == "sigmoid"
    assert artifact.threshold == 0.5 and artifact.n_labels == 6


def test_head_lookup_unknown_name_raises():
    card = load_model_card_from_dict(V2_CARD)
    with pytest.raises(ModelCardError):
        card.head("nope")


def test_v2_ordinal_class_order_must_match_top_level():
    bad = json.loads(json.dumps(V2_CARD))
    bad["heads"][0]["class_order"] = ["Q0", "Q1", "Q2"]  # != top-level 4-class
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(bad)


def test_v2_no_ordinal_head_raises():
    bad = json.loads(json.dumps(V2_CARD))
    bad["heads"] = [bad["heads"][1], bad["heads"][2]]  # drop the ordinal head
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(bad)


def test_v2_multilabel_requires_valid_threshold():
    missing = json.loads(json.dumps(V2_CARD))
    del missing["heads"][2]["threshold"]
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(missing)
    out_of_range = json.loads(json.dumps(V2_CARD))
    out_of_range["heads"][2]["threshold"] = 1.5
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(out_of_range)


def test_v2_binary_head_needs_two_classes():
    bad = json.loads(json.dumps(V2_CARD))
    bad["heads"][1]["class_order"] = ["BAD", "MEH", "OK"]
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(bad)


def test_v2_unknown_kind_or_activation_raises():
    bad_kind = json.loads(json.dumps(V2_CARD))
    bad_kind["heads"][0]["kind"] = "regression"
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(bad_kind)
    bad_act = json.loads(json.dumps(V2_CARD))
    bad_act["heads"][2]["activation"] = "softmax"  # multilabel must be sigmoid
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(bad_act)


def test_v2_duplicate_output_name_raises():
    bad = json.loads(json.dumps(V2_CARD))
    bad["heads"][1]["output_name"] = "q_logits"  # collides with the grade head
    with pytest.raises(ModelCardError):
        load_model_card_from_dict(bad)


def test_validate_onnx_input_shape_accepts_dynamic_batch():
    card = load_model_card_from_dict(VALID_CARD)
    validate_onnx_input_shape(card, ("batch", 1, 2048))  # symbolic batch
    validate_onnx_input_shape(card, (None, 1, 2048))     # unknown batch
    validate_onnx_input_shape(card, (-1, 1, 2048))       # dynamic batch sentinel


def test_validate_onnx_input_shape_rejects_wrong_channel_or_rank():
    card = load_model_card_from_dict(VALID_CARD)
    with pytest.raises(ModelCardError):
        validate_onnx_input_shape(card, (1, 2, 2048))       # channel != 1
    with pytest.raises(ModelCardError):
        validate_onnx_input_shape(card, (1, 1, 2048, 1))    # wrong rank
