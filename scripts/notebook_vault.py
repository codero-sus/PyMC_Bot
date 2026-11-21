#!/usr/bin/env python3
"""Encrypt or decrypt an explicitly chosen file with a private AES-256-GCM-SIV key file.

The envelope carries the nonce. Fixed-nonce encryption leaks equality; never use the
same key and nonce with ordinary AES-GCM. The key file must never be committed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCMSIV

MAGIC = b"PYMC-SIV-1\0"


def material(path: Path) -> tuple[bytes, bytes]:
    values = json.loads(path.read_text(encoding="utf-8"))
    key = base64.b64decode(values["key_base64"], validate=True)
    nonce = base64.b64decode(values["iv_base64"], validate=True)
    if len(key) != 32 or len(nonce) != 12:
        raise ValueError("key must be 32 bytes and IV must be 12 bytes")
    return key, nonce


def encrypt(data: bytes, key: bytes, nonce: bytes) -> bytes:
    return MAGIC + nonce + AESGCMSIV(key).encrypt(nonce, data, MAGIC)


def decrypt(data: bytes, key: bytes, expected_nonce: bytes) -> bytes:
    if not data.startswith(MAGIC) or len(data) < len(MAGIC) + 28:
        raise ValueError("not a PyMC AES-GCM-SIV envelope")
    nonce = data[len(MAGIC):len(MAGIC) + 12]
    if nonce != expected_nonce:
        raise ValueError("envelope IV does not match the key file")
    return AESGCMSIV(key).decrypt(nonce, data[len(MAGIC) + 12:], MAGIC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("encrypt", "decrypt"))
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    key, nonce = material(args.key_file)
    source = args.source.read_bytes()
    result = encrypt(source, key, nonce) if args.mode == "encrypt" else decrypt(source, key, nonce)
    # O_EXCL avoids overwriting plaintext, ciphertext, or a key by mistake.
    fd = os.open(args.destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(result)
    print(f"{args.mode} complete: {args.destination}")


if __name__ == "__main__":
    main()
