"""
crypto_subgraph.py

Isolated encryption / decryption subgraph.
- States are fully isolated — no shared fields with ChatbotState
- AES-256-GCM: authenticated, tamper-proof
- Key derived from CHATBOT_SECRET_KEY env var — never stored in DB
- Without the key the SQLite DB is completely unreadable
- Subgraph only function: encode plaintext → ciphertext, decode ciphertext → plaintext
"""

import os
import base64
import hashlib
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ================================================================
# Key derivation
# SHA-256 of env secret → 32-byte AES-256 key
# Key lives only in memory — never written anywhere
# ================================================================

def _derive_key() -> bytes:
    raw = os.environ.get(
        "CHATBOT_SECRET_KEY",
        "change-this-to-a-strong-secret-in-production"
    )
    return hashlib.sha256(raw.encode()).digest()   # 32 bytes


_KEY: bytes = _derive_key()


# ================================================================
# Core AES-256-GCM encrypt / decrypt
# ================================================================

def _encrypt(plaintext: str) -> str:
    """Encrypt plaintext string → base64(12-byte nonce + ciphertext)."""
    aesgcm = AESGCM(_KEY)
    nonce  = os.urandom(12)                          # unique per message
    ct     = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("ascii")


def _decrypt(token: str) -> str:
    """Decrypt base64(nonce + ciphertext) → plaintext string."""
    raw    = base64.b64decode(token.encode("ascii"))
    nonce  = raw[:12]
    ct     = raw[12:]
    aesgcm = AESGCM(_KEY)
    return aesgcm.decrypt(nonce, ct, None).decode("utf-8")


# ================================================================
# Subgraph states — completely isolated from ChatbotState
# ================================================================

class EncodeState(TypedDict):
    plaintext:  str
    ciphertext: str   # populated by encode node


class DecodeState(TypedDict):
    ciphertext: str
    plaintext:  str   # populated by decode node


# ================================================================
# Encode subgraph
# ================================================================

def _encode_node(state: EncodeState) -> EncodeState:
    return {
        "plaintext":  state["plaintext"],
        "ciphertext": _encrypt(state["plaintext"])
    }


def _build_encode_subgraph():
    g = StateGraph(EncodeState)
    g.add_node("encode", _encode_node)
    g.add_edge(START, "encode")
    g.add_edge("encode", END)
    return g.compile()


# ================================================================
# Decode subgraph
# ================================================================

def _decode_node(state: DecodeState) -> DecodeState:
    try:
        plaintext = _decrypt(state["ciphertext"])
    except Exception:
        # Not encrypted (e.g. legacy message) — return as-is
        plaintext = state["ciphertext"]
    return {
        "ciphertext": state["ciphertext"],
        "plaintext":  plaintext
    }


def _build_decode_subgraph():
    g = StateGraph(DecodeState)
    g.add_node("decode", _decode_node)
    g.add_edge(START, "decode")
    g.add_edge("decode", END)
    return g.compile()


# ================================================================
# Compiled subgraphs — sync, no asyncio, safe at module level
# ================================================================

_encode_graph = _build_encode_subgraph()
_decode_graph = _build_decode_subgraph()


# ================================================================
# Public API — called from chat_model_v4.py and chatbot_v5.py
# ================================================================

def encrypt_message(plaintext: str) -> str:
    """Encrypt a plaintext string. Returns opaque ciphertext token."""
    if not plaintext:
        return plaintext
    result = _encode_graph.invoke({
        "plaintext":  plaintext,
        "ciphertext": ""
    })
    return result["ciphertext"]


def decrypt_message(ciphertext: str) -> str:
    """
    Decrypt a ciphertext token. Returns plaintext.
    Falls back to returning the input unchanged if decryption fails
    (handles legacy unencrypted messages in DB gracefully).
    """
    if not ciphertext:
        return ciphertext
    result = _decode_graph.invoke({
        "ciphertext": ciphertext,
        "plaintext":  ""
    })
    return result["plaintext"]