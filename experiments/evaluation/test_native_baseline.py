"""Offline checks for independently approved native dialog scope."""
import json

import pytest

from experiments.evaluation import native_baseline
from experiments.evaluation import native_agent_baseline


@pytest.mark.parametrize("value", [None, "", "replace-with-private-id"])
@pytest.mark.parametrize("module", [native_baseline, native_agent_baseline])
def test_missing_chat_approval_stops_prepare_before_runtime_access(monkeypatch, tmp_path, value, module):
    if value is None:
        monkeypatch.delenv("POLICY_APPROVED_CHAT_ID", raising=False)
    else:
        monkeypatch.setenv("POLICY_APPROVED_CHAT_ID", value)
    monkeypatch.setattr(native_baseline, "OUT", tmp_path / "baseline")
    with pytest.raises(ValueError, match="POLICY_APPROVED_CHAT_ID"):
        module.prepare() if module is native_baseline else module.prepare(tmp_path / "agent")
    assert not (tmp_path / "baseline").exists()
    assert not (tmp_path / "agent").exists()


@pytest.mark.parametrize("module", [native_baseline, native_agent_baseline])
def test_frozen_chat_cannot_authorize_itself(monkeypatch, tmp_path, module):
    monkeypatch.setenv("POLICY_APPROVED_CHAT_ID", "1" * 32)
    name = "prepared.json" if module is native_baseline else "config.json"
    (tmp_path / name).write_text(json.dumps({"chat_id": "2" * 32}))
    monkeypatch.setattr(native_baseline, "OUT", tmp_path)
    with pytest.raises(PermissionError, match="approved"):
        module.run("preflight", "attempt") if module is native_baseline else module.run(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [name]
