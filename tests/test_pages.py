import pytest
from streamlit.testing.v1 import AppTest

from src.data_processing.loader import PROJECT_ROOT, SAMPLE_PATH

PAGE = str(PROJECT_ROOT / "pages" / "02_Dataset_Explorer.py")
needs_sample = pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="demo sample not present")


@needs_sample
def test_explorer_opens_in_demo_mode_without_errors():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Records" in labels and "Failure rate" in labels


@needs_sample
def test_explorer_upload_mode_waits_for_a_file():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Upload my own CSV").run()
    assert not at.exception
    assert any("Upload a CSV" in i.value for i in at.info)