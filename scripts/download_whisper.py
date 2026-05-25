# one-time download — keeps whisper off the broken HF cache on Windows

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "whisper-tiny"


def main():
    from huggingface_hub import snapshot_download

    TARGET.mkdir(parents=True, exist_ok=True)
    print("downloading faster-whisper-tiny ...")
    snapshot_download("Systran/faster-whisper-tiny", local_dir=str(TARGET))
    if not (TARGET / "model.bin").exists():
        raise SystemExit("download failed — model.bin missing")
    print("ok. run: python run_benchmark.py")


if __name__ == "__main__":
    main()
