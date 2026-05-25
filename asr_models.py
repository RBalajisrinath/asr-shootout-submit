# Wrappers for Deepgram + faster-whisper (two language settings)

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

_whisper = None
ROOT = Path(__file__).resolve().parent
WHISPER_DIR = ROOT / "models" / "whisper-tiny"


def whisper_path():
    p = os.environ.get("WHISPER_MODEL_PATH", "").strip()
    if p and Path(p).exists():
        return p
    if (WHISPER_DIR / "model.bin").exists():
        return str(WHISPER_DIR)
    return os.environ.get("WHISPER_MODEL", "tiny")


def load_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    from faster_whisper import WhisperModel

    path = whisper_path()
    print(f"  loading whisper from {path} ...", flush=True)
    use_gpu = os.environ.get("USE_CUDA", "").lower() in ("1", "true")
    device = "cuda" if use_gpu else "cpu"
    dtype = "float16" if device == "cuda" else "int8"
    _whisper = WhisperModel(path, device=device, compute_type=dtype)
    return _whisper


def deepgram(path: Path, api_key: str) -> dict:
    from deepgram import DeepgramClient

    client = DeepgramClient(api_key=api_key)
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        blob = f.read()
    resp = client.listen.v1.media.transcribe_file(
        request=blob,
        model="nova-2",
        language="en-IN",
        smart_format=True,
        punctuate=True,
    )
    took = time.perf_counter() - t0
    try:
        text = resp.results.channels[0].alternatives[0].transcript
    except Exception:
        text = ""
    return {"text": (text or "").strip(), "latency_sec": round(took, 3)}


def whisper_auto(path: Path) -> dict:
    model = load_whisper()
    t0 = time.perf_counter()
    segs, info = model.transcribe(str(path), language=None, vad_filter=True)
    text = " ".join(s.text.strip() for s in segs).strip()
    took = time.perf_counter() - t0
    return {
        "text": text,
        "latency_sec": round(took, 3),
        "detected_language": getattr(info, "language", None),
    }


def whisper_hi(path: Path) -> dict:
    model = load_whisper()
    t0 = time.perf_counter()
    segs, _ = model.transcribe(str(path), language="hi", vad_filter=True)
    text = " ".join(s.text.strip() for s in segs).strip()
    took = time.perf_counter() - t0
    return {"text": text, "latency_sec": round(took, 3)}


MODELS = {
    "deepgram": "Deepgram nova-2 en-IN",
    "whisper_open": "Whisper tiny, auto language",
    "whisper_hindi": "Whisper tiny, Hindi forced",
}


def run_asr(name: str, path: Path, api_key: str | None) -> dict:
    try:
        if name == "deepgram":
            if not api_key:
                return {"text": "", "latency_sec": 0, "error": "no DEEPGRAM_API_KEY in .env"}
            return deepgram(path, api_key)
        if name == "whisper_open":
            return whisper_auto(path)
        if name == "whisper_hindi":
            return whisper_hi(path)
        raise ValueError(name)
    except Exception as e:
        return {"text": "", "latency_sec": 0, "error": str(e)}
