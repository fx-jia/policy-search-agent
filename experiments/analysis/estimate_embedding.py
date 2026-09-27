"""Count local cl100k tokens as a cost proxy, with all socket access disabled.

Consumes a JSON object on stdin containing documents and questions. Run with the
already cached tiktoken data in the RAGFlow container. No provider SDK is loaded.
These are not provider billing tokens or actual parser chunk counts.
"""

import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def deny_network(*args, **kwargs):
    raise RuntimeError("Network disabled during offline token estimation")


def main():
    socket.socket = deny_network
    socket.create_connection = deny_network
    cache = Path(os.environ["TIKTOKEN_CACHE_DIR"])
    cache_key = hashlib.sha1(b"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken").hexdigest()
    if not (cache / cache_key).is_file():
        raise ValueError("An existing local cl100k tokenizer cache is required")
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    data = json.load(sys.stdin)

    def count(text):
        return len(encoder.encode(text, disallowed_special=()))

    documents = [{"source_document_id": row["id"], "text_tokens": count(row["text"]), "filename_tokens": count(row["upload_name"]), "unicode_chars": len(row["text"])} for row in data["documents"]]
    questions = [{"question_id": row["id"], "tokens": count(row["text"])} for row in data["questions"]]
    json.dump(
        {
            "tokenizer": "cl100k_base",
            "tiktoken_version": tiktoken.__version__,
            "tokenizer_cache_sha256": hashlib.sha256((cache / cache_key).read_bytes()).hexdigest(),
            "basis": "unparsed_markdown_plus_one_filename_embedding_per_document",
            "provider_usage_measured": False,
            "network_disabled": True,
            "documents": documents,
            "questions": questions,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    print()


if __name__ == "__main__":
    main()
