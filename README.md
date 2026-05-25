# ASR shootout

I recorded 20 phone-style clips with Bangalore locality names (Tanglish/English) and compared Deepgram to two Whisper setups. Full write-up is in `REPORT.md`.

**Balajisrinath R**

## Run it

```powershell
cd asr-shootout
copy .env.example .env
# add DEEPGRAM_API_KEY

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_whisper.py
python run_benchmark.py
```

First time, Whisper downloads ~75 MB. Output goes to `results/`; charts to `figures/`.

## Models

- **deepgram** — nova-2, Indian English (baseline)
- **whisper_open** — tiny, auto language
- **whisper_hindi** — same model, Hindi forced

Put your 20 files in `recordings/` — names must match `references.json` (e.g. `001_koramangala_quiet.mp4`).
