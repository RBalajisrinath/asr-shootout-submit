# Run all ASR models on recordings/, score vs references.json, write results/

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from asr_models import MODELS, run_asr
from metrics import score_sample

ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"
RESULTS = ROOT / "results"
REFS = ROOT / "references.json"
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".webm", ".flac"}

# rough Deepgram pay-as-you-go $/min for the cost table
USD_PER_MIN = {"deepgram": 0.0043, "whisper_open": 0.0, "whisper_hindi": 0.0}


def load_refs():
    with open(REFS, encoding="utf-8") as f:
        return json.load(f)


def audio_files():
    return sorted(p for p in RECORDINGS.iterdir() if p.suffix.lower() in AUDIO_EXT)


def clip_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


def length_tag(sec: float) -> str:
    if sec <= 0:
        return "unknown"
    if sec < 4:
        return "short"
    if sec < 8:
        return "medium"
    return "long"


def total_minutes(paths: list[Path]) -> float:
    try:
        import soundfile as sf

        return sum(sf.info(str(p)).duration for p in paths) / 60.0
    except Exception:
        return len(paths) * 5 / 60  # guess ~5s each


def main():
    load_dotenv(ROOT / ".env")
    key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    refs = load_refs()
    files = audio_files()
    if not files:
        print("put audio in recordings/ first")
        sys.exit(1)

    RESULTS.mkdir(exist_ok=True)
    only = os.environ.get("ONLY_MODELS", "").strip()
    models = [m.strip() for m in only.split(",") if m.strip()] if only else list(MODELS)

    rows = []
    for path in files:
        stem = path.stem
        if stem not in refs:
            print(f"skip {path.name} (not in references.json)")
            continue
        meta = refs[stem]
        dur = clip_seconds(path)
        for model in models:
            print(f"  {model}: {path.name}")
            out = run_asr(model, path, key)
            hyp = out.get("text", "")
            scores = score_sample(meta["reference"], hyp, meta["locality"])
            rows.append(
                {
                    "file": path.name,
                    "stem": stem,
                    "locality": meta["locality"],
                    "condition": meta.get("condition"),
                    "language": meta.get("language"),
                    "length": length_tag(dur),
                    "duration_sec": round(dur, 2),
                    "model": model,
                    "reference": meta["reference"],
                    "hypothesis": hyp,
                    "latency_sec": out.get("latency_sec"),
                    "error": out.get("error"),
                    **scores,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_sample_scores.csv", index=False)

    summary = df.groupby("model").agg(
        wer_mean=("wer", "mean"),
        cer_mean=("cer", "mean"),
        locality_acc=("locality_matched", "mean"),
        latency_mean=("latency_sec", "mean"),
        n=("file", "count"),
    )
    summary = summary.reset_index()
    dg = summary[summary["model"] == "deepgram"]
    if len(dg):
        summary["wer_vs_dg"] = summary["wer_mean"] - float(dg["wer_mean"].iloc[0])
        summary["loc_vs_dg"] = summary["locality_acc"] - float(dg["locality_acc"].iloc[0])
    summary.to_csv(RESULTS / "summary.csv", index=False)

    for col, out_name in [
        ("condition", "by_condition.csv"),
        ("language", "by_language.csv"),
        ("length", "by_length.csv"),
    ]:
        g = df.groupby(["model", col]).agg(
            wer_mean=("wer", "mean"),
            locality_acc=("locality_matched", "mean"),
        )
        g.reset_index().to_csv(RESULTS / out_name, index=False)

    # clips where challenger got locality but deepgram didn't
    dg_loc = df[df["model"] == "deepgram"][["stem", "locality_matched"]].rename(
        columns={"locality_matched": "dg_ok"}
    )
    h2h = []
    for model in models:
        if model == "deepgram":
            continue
        m = df[df["model"] == model].merge(dg_loc, on="stem")
        wins = ((m["locality_matched"]) & (~m["dg_ok"])).sum()
        losses = ((~m["locality_matched"]) & (m["dg_ok"])).sum()
        h2h.append({"model": model, "wins": int(wins), "losses": int(losses)})
    pd.DataFrame(h2h).to_csv(RESULTS / "head_to_head.csv", index=False)

    mins = total_minutes(files)
    pd.DataFrame(
        [
            {
                "model": m,
                "minutes": round(mins, 2),
                "est_usd_10k_min": round(USD_PER_MIN.get(m, 0) * 10000, 1),
            }
            for m in models
        ]
    ).to_csv(RESULTS / "cost_estimate.csv", index=False)

    make_plots(summary, df)
    print("\n", summary[["model", "wer_mean", "locality_acc", "latency_mean"]])
    print("done -> results/")


def make_plots(summary, df):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    models = summary["model"].tolist()
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, summary["locality_acc"], 0.4, label="locality match")
    ax.bar(x + 0.2, 1 - summary["wer_mean"].clip(0, 1), 0.4, label="1 - WER")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_title("locality vs full sentence")
    plt.tight_layout()
    fig.savefig(fig_dir / "metrics_comparison.png", dpi=120)
    plt.close()

    pivot = df.pivot_table(index="condition", columns="model", values="locality_matched", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(fig_dir / "condition_heatmap.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
