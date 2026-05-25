# WER/CER + did we get the locality name out of the transcript?

from __future__ import annotations

import re
import unicodedata
from typing import Any

import jiwer
from rapidfuzz import fuzz


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return float(jiwer.wer(r, h))


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return float(jiwer.cer(r, h))


def locality_aliases(name: str) -> list[str]:
    # spellings / shortenings that still count as a match
    base = normalize(name)
    out = {base, base.replace(" ", "")}
    if "layout" in base:
        out.add(base.replace("layout", "").strip())
    if "puram" in base and "kr" in base:
        out.update({"kr puram", "krishnarajapuram"})
    if "electronic" in base:
        out.add("electronic city")
    if "silk" in base and "board" in base:
        out.add("silkboard")
    return [x for x in out if x]


def locality_match(name: str, transcript: str, cutoff: int = 82) -> dict[str, Any]:
    hyp = normalize(transcript)
    best = 0
    hit = False
    for alias in locality_aliases(name):
        if alias in hyp:
            return {"matched": True, "best_score": 100, "best_variant": alias}
        score = fuzz.partial_ratio(alias, hyp)
        if score > best:
            best = score
    if best >= cutoff:
        hit = True
    return {"matched": hit, "best_score": best, "best_variant": ""}


def score_sample(reference: str, hypothesis: str, locality: str) -> dict[str, Any]:
    loc = locality_match(locality, hypothesis)
    return {
        "wer": wer(reference, hypothesis),
        "cer": cer(reference, hypothesis),
        "locality_matched": loc["matched"],
        "locality_fuzzy_score": loc["best_score"],
        "locality_wer": 0.0 if loc["matched"] else 1.0,
    }
