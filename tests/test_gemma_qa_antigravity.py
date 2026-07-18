"""Antigravity path retired in favor of TNG three-model stack."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Antigravity retired; TNG three-model stack")


def test_antigravity_retired_placeholder() -> None:
    assert True
