"""Every test reads afresh, its reads kept in its own folder: none left in the machine's."""

import pytest


@pytest.fixture(autouse=True)
def _kept_apart(tmp_path, monkeypatch):
    from usdscope import kept, read

    monkeypatch.setattr(kept, "KEPT", tmp_path / "reads")
    monkeypatch.setattr(read, "COLOURS", tmp_path / "colours.json")
