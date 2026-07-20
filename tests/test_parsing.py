"""MedGemma output parsing — must never raise, must survive messy real output.

These test the pure, torch-free parsing surface (`anotmed/backends/parsing.py`)
that turns a raw model string into Detections. The model output is adversarial
input: it can contain prose, code fences, brackets inside labels, non-numeric
coordinates, and out-of-range values. A parser that raises on any of these would
crash the whole study upload, so the contract is: skip the bad element, keep the
good ones, and on total failure return [].
"""

from __future__ import annotations

import pytest

from anotmed.backends.parsing import extract_json_array, parse_detections


# ---- extract_json_array: string-aware, fence-tolerant -----------------------

def test_extract_survives_unbalanced_bracket_inside_label():
    # A naive bracket-depth counter decrements on the ']' inside "lesion ]" and
    # bails early. A string-aware decoder must keep the whole array.
    text = '[{"label": "lesion ]", "box_2d": [10, 20, 30, 40]}]'
    result = extract_json_array(text)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["label"] == "lesion ]"


def test_extract_ignores_leading_prose_and_stray_bracket():
    text = 'Sure [see note]. Findings: [{"label": "nodule", "box_2d": [1,2,3,4]}]'
    result = extract_json_array(text)
    assert len(result) == 1
    assert result[0]["label"] == "nodule"


def test_extract_tolerates_code_fence():
    text = '```json\n[{"label": "mass", "box_2d": [0,0,100,100]}]\n```'
    result = extract_json_array(text)
    assert len(result) == 1


def test_extract_empty_string_returns_empty_list():
    assert extract_json_array("") == []


def test_extract_no_array_returns_empty_list():
    assert extract_json_array("The scan is unremarkable.") == []


def test_extract_garbage_returns_empty_list():
    assert extract_json_array("[[[[ not json at all") == []


# ---- parse_detections: per-element resilience, clamping, score default ------

def test_parse_one_bad_box_of_three_keeps_the_two_good_ones():
    text = (
        '[{"label": "a", "box_2d": [10, 10, 20, 20]},'
        ' {"label": "b", "box_2d": ["x", 10, 20, 20]},'  # non-numeric coord
        ' {"label": "c", "box_2d": [30, 30, 40, 40]}]'
    )
    dets = parse_detections(text, rows=100, cols=100)
    assert [d.label for d in dets] == ["a", "c"]


def test_parse_clamps_out_of_range_coords_into_the_image():
    text = '[{"label": "big", "box_2d": [-100, -100, 3000, 3000]}]'
    dets = parse_detections(text, rows=100, cols=200)
    assert len(dets) == 1
    b = dets[0].box
    assert b.x >= 0 and b.y >= 0
    assert b.x + b.w <= 200
    assert b.y + b.h <= 100


def test_parse_default_score_is_one_so_min_score_does_not_nuke_findings():
    # MedGemma emits no score; a 0.0 default would be filtered out the moment an
    # owner sets ANOTMED_MIN_SCORE > 0. See pipeline.py:87.
    text = '[{"label": "nodule", "box_2d": [10, 10, 20, 20]}]'
    dets = parse_detections(text, rows=100, cols=100)
    assert dets[0].score == 1.0


def test_parse_honors_explicit_score_when_present():
    text = '[{"label": "nodule", "box_2d": [10, 10, 20, 20], "score": 0.4}]'
    dets = parse_detections(text, rows=100, cols=100)
    assert dets[0].score == pytest.approx(0.4)


def test_parse_skips_wrong_length_box():
    text = '[{"label": "bad", "box_2d": [10, 20, 30]}]'  # only 3 coords
    assert parse_detections(text, rows=100, cols=100) == []


def test_parse_skips_non_dict_items():
    text = '["just a string", 42, {"label": "ok", "box_2d": [1,2,3,4]}]'
    dets = parse_detections(text, rows=100, cols=100)
    assert [d.label for d in dets] == ["ok"]


def test_parse_missing_label_defaults_to_finding():
    text = '[{"box_2d": [10, 10, 20, 20]}]'
    dets = parse_detections(text, rows=100, cols=100)
    assert dets[0].label == "finding"


def test_parse_never_raises_on_pathological_input():
    pathological = [
        "", "[", "]", "[{", "not json", "```json\n[]\n```", "[[[[[",
        '[{"box_2d":[1,2]}]', "\x00\x01\x02", "[{}]", '[{"box_2d":"nope"}]',
        '[null, true, {"box_2d": null}]', '[{"box_2d": [1, 2, 3, "z"]}]',
    ]
    for text in pathological:
        result = parse_detections(text, rows=64, cols=64)
        assert isinstance(result, list)
