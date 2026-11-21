import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def test_dashboard_runs():
    path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(path, default_timeout=120)
    at.run()
    assert not at.exception
