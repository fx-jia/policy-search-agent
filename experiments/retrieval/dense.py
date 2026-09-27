"""Strict KNN adapter for the user's existing local RAGFlow container."""

import json
import hashlib
from pathlib import Path
import select
import subprocess

from experiments.retrieval.ragflow import RetrievalError, build_payload, parse_response


class DenseClient:
    def __init__(self, config, mapping):
        if config["method"] != "dense" or config.get("rerank_model") or config["retrieval_config"]["vector_similarity_weight"] != 1:
            raise ValueError("Strict Dense uses only vector scoring and no reranker")
        # Validate the shared candidate/threshold fields, without sending this payload.
        build_payload("configuration check", dict(config, method="hybrid"))
        if config["base_url"] != "http://127.0.0.1:9380":
            raise ValueError("Dense worker only supports the verified local RAGFlow instance")
        self.config, self.document_map = config, mapping["documents"]
        self.last_stage_latency_ms = None
        self.last_embedding_tokens = None
        self.failed = False
        source = Path(__file__).with_name("dense_worker.py").read_text()
        self.process = subprocess.Popen(
            ["docker", "exec", "-i", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-u", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        try:
            response = self._request({"config": config, "documents": list(self.document_map), "indexed_chunks": mapping["indexed_chunks"]}, timeout=60)
            if response.get("status") != "ready":
                raise ValueError(f"Dense worker initialization failed: {response.get('error_type', 'unknown')}")
        except Exception:
            self.close()
            raise

    def _request(self, payload, timeout=30):
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], timeout)[0]:
                raise RetrievalError("dense_worker_timeout")
            line = self.process.stdout.readline()
            response = json.loads(line)
        except (BrokenPipeError, OSError, ValueError):
            raise RetrievalError("dense_worker_transport_error") from None
        return response

    def retrieve(self, question, limit):
        if limit != self.config["retrieval_config"]["candidate_count"]:
            raise ValueError("Candidate window mismatch")
        self.last_stage_latency_ms = None
        self.last_embedding_tokens = None
        if self.failed:
            raise RetrievalError("dense_worker_unavailable")
        request_id = hashlib.sha256(question.encode()).hexdigest()
        try:
            response = self._request({"question": question, "request_id": request_id})
            if response.get("request_id") != request_id:
                raise RetrievalError("dense_worker_response_mismatch")
        except RetrievalError:
            # A delayed response must never be scored as the next question.
            self.failed = True
            self.close()
            raise
        chunks = parse_response(response, self.document_map)
        self.last_stage_latency_ms = response["stage_latency_ms"]
        self.last_embedding_tokens = response["embedding_tokens"]
        return chunks

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
