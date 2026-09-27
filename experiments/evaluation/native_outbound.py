"""Fail-closed review gate for native chat knowledge sent to a model."""

from copy import deepcopy
import hashlib


def _sha256(value):
    if not isinstance(value, str):
        raise ValueError("Outbound review field is not text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_kbinfos(kbinfos, review, document_map):
    """Replace reviewed chunk text while preserving retrieval order and scores.

    All material rendered by RAGFlow's ``kb_prompt`` must match the frozen
    review. A mismatch aborts before any knowledge is sent to a model.
    """
    if not isinstance(kbinfos, dict) or not isinstance(kbinfos.get("chunks"), list):
        raise ValueError("Invalid retrieval result")
    if not isinstance(review, dict) or not isinstance(document_map, dict):
        raise ValueError("Invalid outbound review inputs")
    sources = document_map.get("documents")
    if not isinstance(sources, dict):
        raise ValueError("Invalid document map")

    sanitized = deepcopy(kbinfos)
    audit_rows = []
    for chunk in sanitized["chunks"]:
        if not isinstance(chunk, dict):
            raise ValueError("Invalid retrieved chunk")
        cid = chunk.get("chunk_id", chunk.get("id"))
        docid = chunk.get("doc_id")
        entry = review.get(cid)
        if not isinstance(cid, str) or not cid or not isinstance(entry, dict):
            raise ValueError("Unreviewed retrieved chunk")
        if not isinstance(docid, str) or sources.get(docid) != entry.get("source_document_id"):
            raise ValueError("Retrieved chunk source mismatch")

        raw = chunk.get("content_with_weight")
        safe = entry.get("sanitized_content")
        if _sha256(raw) != entry.get("raw_sha256") or _sha256(safe) != entry.get("safe_sha256"):
            raise ValueError("Retrieved chunk content or review hash mismatch")

        title = chunk.get("docnm_kwd")
        raw_title = entry.get("raw_title")
        safe_title = entry.get("safe_title")
        if (
            not isinstance(title, str)
            or title != raw_title
            or _sha256(title) != entry.get("raw_title_sha256")
            or _sha256(safe_title) != entry.get("safe_title_sha256")
        ):
            raise ValueError("Retrieved chunk title or review hash mismatch")

        # These aliases can take precedence in RAGFlow's prompt renderer.
        # No alias, URL, or document metadata is approved in this baseline.
        for field in ("content", "document_name", "url", "document_metadata"):
            if chunk.get(field):
                raise ValueError("Unreviewed prompt field")
            chunk.pop(field, None)

        chunk["content_with_weight"] = safe
        chunk["docnm_kwd"] = safe_title
        audit_rows.append(
            {
                "chunk_id": cid,
                "source_document_id": entry["source_document_id"],
                "raw_sha256": entry["raw_sha256"],
                "safe_sha256": entry["safe_sha256"],
                "changed": raw != safe or title != safe_title,
            }
        )

    return sanitized, {"chunks": audit_rows, "total": len(audit_rows), "changed": sum(row["changed"] for row in audit_rows)}
