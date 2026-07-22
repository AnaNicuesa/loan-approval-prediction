import time

from src.predict import build_input_frame
from src.utils import CATEGORICAL_FEATURES, EXAMPLE_APPLICANTS, NUMERIC_FEATURES, timer


def test_timer_measures_elapsed_time():
    with timer() as t:
        time.sleep(0.01)
    assert t["seconds"] >= 0.01


def test_timer_reports_none_before_block_finishes():
    with timer() as t:
        assert t["seconds"] is None


def test_numeric_and_categorical_feature_lists_do_not_overlap():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_example_applicants_are_valid_build_input_frame_arguments():
    for name, applicant in EXAMPLE_APPLICANTS.items():
        frame = build_input_frame(**applicant)
        assert len(frame) == 1, f"{name} did not build a single-row frame"
