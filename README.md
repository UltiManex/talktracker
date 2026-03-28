# U Talk Too Much

**U Talk Too Much** is a speaker-aware conversation analytics prototype that:

- Transcribes conversations from audio.
- Attributes transcript segments to individual speakers.
- Calculates who spoke the most (percentage share).
- Displays a simple pie chart by speaker.
- Summarizes the overall discussion.
- Extracts each speaker's main focus keywords.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

## How it works

1. Audio is transcribed with `faster-whisper`.
2. Transcript segments are speaker-labeled (currently with a deterministic mock diarization helper).
3. The app computes speaking-time percentages.
4. Results are rendered as a pie chart plus summaries.

## Replacing mock diarization (production)

The current prototype uses `mock_diarization` in `app.py`. For production-grade real-time performance, plug in one of the following:

- `pyannote.audio` diarization pipeline.
- A managed realtime API such as Deepgram or AssemblyAI with speaker diarization.

Then feed diarization segments into `ConversationAnalyzer.assign_speakers`.

## Testing

```bash
pytest
```
