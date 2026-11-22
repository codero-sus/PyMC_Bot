#!/usr/bin/env python3
"""Protect a notebook vault key with a locally entered passphrase.

The encrypted envelope can be shared; its confidentiality depends on a strong,
unique passphrase. Never share the passphrase or decrypted key. Run locally.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from notebook_vault import material

MAGIC = b"PYMC-PASSPHRASE-KEY-1"
MAX_BYTES = 16_384


def _derive(passphrase: str, salt: bytes) -> bytes:
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    # Fixed KDF parameters are authenticated through the versioned format, not supplied by the file.
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode("utf-8"))


def seal(key: bytes, iv: bytes, passphrase: str) -> bytes:
    if len(key) != 32 or len(iv) != 12:
        raise ValueError("notebook key must be 32 bytes and IV must be 12 bytes")
    payload = json.dumps({"algorithm": "AES-256-GCM-SIV", "key_base64": base64.b64encode(key).decode(),
                          "iv_base64": base64.b64encode(iv).decode()}, separators=(",", ":")).encode()
    salt, nonce = os.urandom(16), os.urandom(12)
    return MAGIC + salt + nonce + AESGCM(_derive(passphrase, salt)).encrypt(nonce, payload, MAGIC)


def unseal(data: bytes, passphrase: str) -> bytes:
    if len(data) > MAX_BYTES or len(data) < len(MAGIC) + 16 + 12 + 16 or not data.startswith(MAGIC):
        raise ValueError("invalid passphrase-protected key envelope")
    offset = len(MAGIC)
    salt, nonce, ciphertext = data[offset:offset + 16], data[offset + 16:offset + 28], data[offset + 28:]
    try:
        plaintext = AESGCM(_derive(passphrase, salt)).decrypt(nonce, ciphertext, MAGIC)
    except InvalidTag as exc:
        raise ValueError("wrong passphrase or damaged key envelope") from exc
    try:
        row = json.loads(plaintext)
        key = base64.b64decode(row["key_base64"], validate=True)
        iv = base64.b64decode(row["iv_base64"], validate=True)
        if row.get("algorithm") != "AES-256-GCM-SIV" or len(key) != 32 or len(iv) != 12:
            raise ValueError("invalid decrypted key")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid decrypted key") from exc
    return plaintext + b"\n"


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    new = sub.add_parser("new", help="create a new passphrase-encrypted vault key")
    new.add_argument("destination", type=Path)
    wrap = sub.add_parser("wrap", help="protect an existing local vault key")
    wrap.add_argument("key_file", type=Path)
    wrap.add_argument("destination", type=Path)
    restore = sub.add_parser("unwrap", help="write a decrypted key file locally")
    restore.add_argument("source", type=Path)
    restore.add_argument("destination", type=Path)
    args = parser.parse_args()
    phrase = getpass.getpass("Passphrase (not echoed): ")
    if args.mode in ("new", "wrap"):
        if phrase != getpass.getpass("Repeat passphrase: "):
            raise SystemExit("Passphrases do not match; nothing written")
        if len(phrase) < 14:
            raise SystemExit("Use a unique passphrase of at least 14 characters; nothing written")
        key, iv = (os.urandom(32), os.urandom(12)) if args.mode == "new" else material(args.key_file)
        result = seal(key, iv, phrase)
    else:
        result = unseal(args.source.read_bytes(), phrase)
    _write_new(args.destination, result)
    print("Encrypted key saved." if args.mode != "unwrap" else "Key restored locally; keep this file private.")


if __name__ == "__main__":
    main()
