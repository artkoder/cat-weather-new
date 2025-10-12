"""Pytest wrapper around the staging E2E script."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import e2e_attach_upload

pytestmark = pytest.mark.e2e


def test_e2e_attach_upload_flow():
    if not os.getenv("E2E_BASE_URL") or not os.getenv("E2E_USER_ID"):
        pytest.skip("E2E environment variables are not configured")
    fname, ctype, data = e2e_attach_upload.make_test_image()
    body, header = e2e_attach_upload.build_multipart(fname, ctype, data)
    assert data and body and "boundary=" in header
    exit_code = e2e_attach_upload.run([])
    assert exit_code == 0
