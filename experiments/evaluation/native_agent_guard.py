"""Durable, fail-closed accounting for an isolated native Agent experiment."""

import json
import hashlib
import os
from pathlib import Path
from threading import RLock


def validate_scope(requested, allowed):
    if not isinstance(requested, (list, tuple)) or not requested or any(not isinstance(x, str) or x not in allowed for x in requested):
        raise ValueError("invalid_or_empty_document_scope")
    return list(requested)


class CallLedger:
    def __init__(self, path, *, run_limit, question_limit, call_limit, identity=None, artifact_root=None):
        self.path = Path(path)
        self.limits = (run_limit, question_limit, call_limit)
        self.lock = RLock()
        self.question_id = None
        self.started = set()
        self.calls = {}
        self.fatal_reason = None
        self._identity = dict(identity) if isinstance(identity, dict) else identity
        self._artifact_root = Path(artifact_root).resolve() if artifact_root is not None else None
        self._digest = hashlib.sha256()
        self._rows = []
        self._reconciliation = None
        if self.path.exists():
            for line in self.path.read_bytes().splitlines(keepends=True):
                self._apply(json.loads(line))
                self._digest.update(line)

    def _apply(self, row):
        event = row["event"]
        if event == "start":
            self.question_id = row["question_id"]
            self.started.add(self.question_id)
            self.fatal_reason = None
        elif event == "reserve":
            self.calls[row["call_id"]] = dict(row)
        elif event == "finish":
            self.calls[row["call_id"]]["usage"] = row["usage"]
            if row.get("transport_status") == "not_sent":
                self.calls[row["call_id"]]["transport_status"] = "not_sent"
                self.calls[row["call_id"]]["transport_error"] = row["transport_error"]
        elif event == "abort":
            self.fatal_reason = row["reason"]
        elif event == "preflight_not_sent_reconciled":
            self._reconciliation = self._validate_preflight_receipt(row)
            self.fatal_reason = None
        else:
            raise ValueError("unknown_ledger_event")
        self._rows.append(row)

    @staticmethod
    def _sha256(value):
        return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)

    def _artifact(self, relative):
        if self._artifact_root is None or not isinstance(relative, str) or not relative:
            raise ValueError("preflight_artifact_context_required")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError("preflight_artifact_must_be_relative")
        resolved = (self._artifact_root / path).resolve()
        if not resolved.is_relative_to(self._artifact_root) or not resolved.is_file():
            raise ValueError("preflight_artifact_outside_session_or_missing")
        return resolved.read_bytes()

    def _matches_identity(self, value):
        return (isinstance(value, dict) and value == self._identity
                and all(type(value[key]) is type(expected) for key, expected in self._identity.items()))

    def _validate_preflight_receipt(self, row):
        expected_rows = [{"event": "start", "question_id": self.question_id},
                         {"event": "abort", "question_id": self.question_id, "reason": "provider_usage_unknown"}]
        if self.calls or self._reconciliation is not None or self.fatal_reason != "provider_usage_unknown" or self._rows != expected_rows:
            raise ValueError("preflight_reconciliation_requires_unreserved_aborted_question")
        if set(row) != {"event", "question_id", "prior_ledger_sha256", "receipt_path", "receipt_sha256"}:
            raise ValueError("invalid_preflight_reconciliation_event")
        identity = self._identity
        if (not isinstance(identity, dict) or set(identity) != {"tenant_id", "kb_id", "session_id", "budget"}
                or any(not isinstance(identity[key], str) or not identity[key].strip() for key in ("tenant_id", "kb_id", "session_id"))
                or type(identity["budget"]) is not int or not 0 < identity["budget"] <= 60000
                or self._artifact_root is None or not self.path.resolve().is_relative_to(self._artifact_root)
                or not self._matches_identity(json.loads(self._artifact("session.json")))):
            raise ValueError("preflight_reconciliation_identity_required")
        prior = self._digest.hexdigest()
        if row.get("question_id") != self.question_id or row.get("prior_ledger_sha256") != prior:
            raise ValueError("preflight_prior_ledger_mismatch")
        receipt_bytes = self._artifact(row.get("receipt_path"))
        if not self._sha256(row.get("receipt_sha256")) or hashlib.sha256(receipt_bytes).hexdigest() != row["receipt_sha256"]:
            raise ValueError("preflight_receipt_hash_mismatch")
        receipt = json.loads(receipt_bytes)
        basis = "source_attested_pre_transport_guard_rejection"
        if (not isinstance(receipt, dict) or receipt.get("schema") != "policy-preflight-not-sent-receipt/v1"
                or receipt.get("question_id") != self.question_id or not self._matches_identity(receipt.get("identity"))
                or receipt.get("original_ledger_sha256") != prior or receipt.get("basis") != basis):
            raise ValueError("preflight_receipt_identity_mismatch")
        author, approver = receipt.get("proof_author"), receipt.get("approved_by")
        if (any(not isinstance(value, str) or not value or value != value.strip() for value in (author, approver))
                or author.casefold() == approver.casefold()):
            raise ValueError("preflight_independent_approval_required")
        proof_bytes = self._artifact(receipt.get("proof_path"))
        if not self._sha256(receipt.get("proof_sha256")) or hashlib.sha256(proof_bytes).hexdigest() != receipt["proof_sha256"]:
            raise ValueError("preflight_proof_hash_mismatch")
        proof = json.loads(proof_bytes)
        if (not isinstance(proof, dict) or proof.get("schema") != "policy-preflight-not-sent-proof/v1"
                or proof.get("question_id") != self.question_id or not self._matches_identity(proof.get("identity"))
                or proof.get("original_ledger_sha256") != prior or proof.get("author") != author or proof.get("basis") != basis):
            raise ValueError("preflight_proof_identity_mismatch")
        facts = {"guard_error": "unbounded_generation_request", "model_name": "qwen3.8-max", "requested_max_tokens": 600,
                 "native_cleaner_deleted_limit": True, "family_restored_limit": False,
                 "offline_outer_attempts": 1, "offline_inner_sends": 0, "setup_model_posts": 0,
                 "streaming_branch": False, "audited_client_only": True, "sdk_max_retries": 0}
        observed = proof.get("facts")
        if (not isinstance(observed, dict) or observed != facts
                or any(type(observed[key]) is not type(value) for key, value in facts.items())):
            raise ValueError("preflight_proof_does_not_establish_guard_rejection")
        hashes = proof.get("source_sha256")
        if (not isinstance(hashes, dict) or set(hashes) != {"native_chat_model", "native_embedding_model", "policy_search_service", "policy_transport", "deployment_manifest", "container_image"}
                or any(not self._sha256(value) for value in hashes.values())):
            raise ValueError("preflight_source_hashes_required")
        report = self._artifact(proof.get("report_path"))
        if not self._sha256(proof.get("report_sha256")) or hashlib.sha256(report).hexdigest() != proof["report_sha256"]:
            raise ValueError("preflight_report_hash_mismatch")
        self._validate_frozen_preflight_evidence(proof, receipt, hashes, prior)
        return {"question_id": self.question_id, "basis": basis, "provider_usage_receipt": False,
                "original_fatal_reason": "provider_usage_unknown", "receipt_path": row["receipt_path"],
                "receipt_sha256": row["receipt_sha256"], "proof_path": receipt["proof_path"],
                "proof_sha256": receipt["proof_sha256"], "proof_author": author, "approved_by": approver}

    def _validate_frozen_preflight_evidence(self, proof, receipt, sources, prior):
        names = {"first-question-deployment.json", "first-question-session.json", "first-question-original-ledger.jsonl", "first-question.json",
                 "native-chat-model.py", "native-embedding-model.py", "native-runtime.json"}
        hashes = proof.get("evidence_sha256")
        if not isinstance(hashes, dict) or set(hashes) != names or any(not self._sha256(value) for value in hashes.values()):
            raise ValueError("preflight_frozen_evidence_required")
        parent = Path(receipt["proof_path"]).parent
        blobs = {}
        for name in names:
            blobs[name] = self._artifact((parent / name).as_posix())
            if hashlib.sha256(blobs[name]).hexdigest() != hashes[name]:
                raise ValueError("preflight_frozen_evidence_hash_mismatch")
        if (hashes["first-question-original-ledger.jsonl"] != prior
                or [json.loads(line) for line in blobs["first-question-original-ledger.jsonl"].splitlines()] != self._rows
                or not self._matches_identity(json.loads(blobs["first-question-session.json"]))):
            raise ValueError("preflight_frozen_ledger_or_session_mismatch")
        question = json.loads(blobs["first-question.json"])
        result, models = question.get("result", {}), question.get("models", {})
        recorded = question.get("ledger", {})
        summary = self.summary()
        if (question.get("question_id") != self.question_id or result.get("session_id") != self._identity["session_id"]
                or result.get("status") != "technical_error" or result.get("stop_reason") != "resolution_error"
                or not str(result.get("error", "")).startswith("Expecting value:")
                or result.get("usage", {}).get("unknown_usage") is not True or question.get("calls") != []
                or models.get("generation") != "qwen3.8-max" or recorded != summary
                or any(type(recorded[key]) is not type(value) for key, value in summary.items())):
            raise ValueError("preflight_frozen_question_mismatch")
        deployment = json.loads(blobs["first-question-deployment.json"])
        scope = deployment.get("scope", {})
        if (sources["deployment_manifest"] != hashes["first-question-deployment.json"]
                or deployment.get("all_file_hashes_verified") is not True
                or any(scope.get(key) != self._identity[key] for key in ("tenant_id", "kb_id"))
                or not self._sha256(deployment.get("container_id"))):
            raise ValueError("preflight_frozen_deployment_mismatch")
        files = deployment.get("files")
        if not isinstance(files, list) or any(not isinstance(row, dict) for row in files):
            raise ValueError("preflight_frozen_deployment_files_invalid")
        for name, target in (("policy_search_service", "/ragflow/api/db/services/policy_search_service.py"),
                             ("policy_transport", "/ragflow/experiments/search_agent/policy_transport.py")):
            rows = [row for row in files if row.get("target") == target]
            if (len(rows) != 1 or rows[0].get("sha256") != sources[name] or rows[0].get("container_sha256") != sources[name]
                    or rows[0].get("verified") is not True or rows[0].get("mode") != "ro"):
                raise ValueError("preflight_frozen_source_mismatch")
        native_hashes = {"native_chat_model": hashes["native-chat-model.py"], "native_embedding_model": hashes["native-embedding-model.py"]}
        runtime = json.loads(blobs["native-runtime.json"])
        if (runtime.get("schema") != "policy-native-runtime-attestation/v1"
                or any(runtime.get(key) != deployment.get(key) for key in ("container_id", "image", "started_at"))
                or not isinstance(deployment.get("image"), str) or not deployment["image"]
                or not isinstance(deployment.get("started_at"), str) or not deployment["started_at"]
                or runtime.get("image_id") != "sha256:" + sources["container_image"]
                or runtime.get("native_source_sha256") != native_hashes
                or any(sources[key] != value for key, value in native_hashes.items())
                or runtime.get("attested_by") != receipt["approved_by"] or runtime.get("report_sha256") != proof["report_sha256"]):
            raise ValueError("preflight_native_runtime_attestation_mismatch")
        limits = {"historical_live_stderr_preserved": False, "historical_packet_trace": False, "provider_usage_receipt": False, "basis_is_source_attested": True}
        if proof.get("limitations") != limits or any(type(proof["limitations"][key]) is not bool for key in limits):
            raise ValueError("preflight_attestation_limitations_required")

    def _reconcile_preflight_not_sent(self, receipt_path):
        """Append independently approved evidence for this exact pre-send failure."""
        with self.lock:
            prior = self.path.read_bytes()
            if not prior.endswith(b"\n") or hashlib.sha256(prior).hexdigest() != self._digest.hexdigest():
                raise ValueError("preflight_prior_ledger_changed")
            row = {"event": "preflight_not_sent_reconciled", "question_id": self.question_id,
                   "prior_ledger_sha256": self._digest.hexdigest(), "receipt_path": receipt_path,
                   "receipt_sha256": hashlib.sha256(self._artifact(receipt_path)).hexdigest()}
            self._validate_preflight_receipt(row)
            self._append(row)

    def _append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(row)
        self._digest.update(line.encode())

    def start(self, question_id):
        with self.lock:
            if not question_id or question_id in self.started:
                raise ValueError("duplicate_or_empty_question")
            if any("usage" not in row for row in self.calls.values()):
                raise ValueError("unresolved_provider_usage")
            self._append({"event": "start", "question_id": question_id})

    def abort(self, reason):
        with self.lock:
            if self.fatal_reason is None:
                self._append({"event": "abort", "question_id": self.question_id, "reason": reason})

    def assert_open(self):
        if not self.question_id:
            raise ValueError("no_active_question")
        if self.fatal_reason:
            raise ValueError(self.fatal_reason)
        if self._reconciliation and self.question_id == self._reconciliation["question_id"]:
            raise ValueError("preflight_question_already_reconciled")

    def reserve(self, call_id, reserve_tokens, request):
        with self.lock:
            self.assert_open()
            if call_id in self.calls or not call_id:
                raise ValueError("duplicate_call_id")
            if not isinstance(reserve_tokens, int) or reserve_tokens <= 0:
                raise ValueError("invalid_reservation")
            current = [r for r in self.calls.values() if r["question_id"] == self.question_id]
            def committed(rows):
                return sum(r["usage"]["total_tokens"] if "usage" in r else r["reserve_tokens"] for r in rows)
            run_limit, question_limit, call_limit = self.limits
            reason = None
            if len(current) >= call_limit:
                reason = "question_call_limit"
            elif committed(self.calls.values()) + reserve_tokens > run_limit:
                reason = "run_token_budget"
            elif committed(current) + reserve_tokens > question_limit:
                reason = "question_token_budget"
            if reason:
                self.abort(reason)
                raise ValueError(reason)
            self._append({"event": "reserve", "question_id": self.question_id, "call_id": call_id, "reserve_tokens": reserve_tokens, "request": request})

    def finish(self, call_id, usage):
        with self.lock:
            row = self.calls.get(call_id)
            if not row or "usage" in row:
                raise ValueError("unknown_or_duplicate_completion")
            if not isinstance(usage, dict) or not isinstance(usage.get("total_tokens"), int) or usage["total_tokens"] <= 0:
                self.abort("provider_usage_unknown")
                raise ValueError("provider_usage_unknown")
            self._append({"event": "finish", "call_id": call_id, "usage": usage})
            if usage["total_tokens"] > row["reserve_tokens"]:
                self.abort("usage_exceeded_reservation")
                raise ValueError("usage_exceeded_reservation")

    def finish_not_sent(self, call_id):
        """Release only an attempt whose physical send raised ConnectError."""
        with self.lock:
            row = self.calls.get(call_id)
            if not row or "usage" in row:
                raise ValueError("unknown_or_duplicate_completion")
            self._append({
                "event": "finish", "call_id": call_id,
                "transport_status": "not_sent", "transport_error": "ConnectError",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    def summary(self):
        with self.lock:
            summary = {
                "provider_calls": len(self.calls),
                "provider_not_sent_calls": sum(r.get("transport_status") == "not_sent" for r in self.calls.values()),
                "provider_total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in self.calls.values()),
                "pending_calls": [cid for cid, row in self.calls.items() if "usage" not in row],
                "fatal_reason": self.fatal_reason,
            }
            if self._reconciliation is not None:
                summary["preflight_not_sent_reconciliation"] = dict(self._reconciliation)
            return summary
