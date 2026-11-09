import io

import pytest

from src.data_processing.loader import read_csv_safely


class FakeUpload(io.BytesIO):
    """BytesIO with a .size attribute, like Streamlit's uploaded files."""

    @property
    def size(self):
        return len(self.getvalue())


def test_reads_a_small_valid_csv():
    df = read_csv_safely(FakeUpload(b"a,b\n1,2\n3,4\n"))
    assert df.shape == (2, 2)


def test_rejects_oversized_file():
    with pytest.raises(ValueError, match="limit"):
        read_csv_safely(FakeUpload(b"a,b\n1,2\n"), max_mb=0)


def test_rejects_too_many_rows():
    data = b"a\n" + b"1\n" * 20
    with pytest.raises(ValueError, match="rows"):
        read_csv_safely(FakeUpload(data), max_rows=10)


def test_rejects_empty_file():
    with pytest.raises(ValueError):
        read_csv_safely(FakeUpload(b""))


def test_rejects_header_only_file():
    with pytest.raises(ValueError, match="no data rows"):
        read_csv_safely(FakeUpload(b"a,b\n"))