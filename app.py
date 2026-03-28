from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import plotly.express as px
import streamlit as st

from utalktoomuch.analyzer import ConversationAnalyzer, SpeakerSegment, TranscriptSegment

st.set_page_config(page_title="U Talk Too Much", page_icon="🗣️", layout="wide")


def mock_diarization(transcript_segments: list[TranscriptSegment], speakers: int) -> list[SpeakerSegment]:
    diarization_segments: list[SpeakerSegment] = []
    for idx, seg in enumerate(transcript_segments):
        speaker = f"Speaker {idx % speakers + 1}"
        diarization_segments.append(SpeakerSegment(speaker=speaker, start=seg.start, end=seg.end))
    return diarization_segments


def transcribe_audio(audio_file: Path) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel

    model = WhisperModel("small", compute_type="int8")
    segments, _ = model.transcribe(str(audio_file), vad_filter=True)

    transcript_segments: list[TranscriptSegment] = []
    for segment in segments:
        transcript_segments.append(
            TranscriptSegment(start=float(segment.start), end=float(segment.end), text=segment.text)
        )
    return transcript_segments


st.title("U Talk Too Much")
st.caption(
    "Realtime-ready conversation analytics prototype: speaker attribution, talk-time percentages, and summaries."
)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_audio = st.file_uploader("Upload a conversation audio file", type=["wav", "mp3", "m4a"])
with col2:
    expected_speakers = st.number_input("Expected speakers", min_value=2, max_value=8, value=2)

if uploaded_audio:
    with NamedTemporaryFile(delete=False, suffix=f"_{uploaded_audio.name}") as tmp:
        tmp.write(uploaded_audio.read())
        tmp_path = Path(tmp.name)

    st.info("Transcribing audio. This may take a few moments.")
    transcript = transcribe_audio(tmp_path)

    analyzer = ConversationAnalyzer()
    diarization = mock_diarization(transcript, expected_speakers)
    labeled_transcript = analyzer.assign_speakers(transcript, diarization)
    shares = analyzer.speaking_share(labeled_transcript)
    summary, speaker_focus = analyzer.summarize(labeled_transcript)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Who talked the most?")
        df = pd.DataFrame(
            [{"Speaker": speaker, "Percent": round(percent, 2)} for speaker, percent in shares.items()]
        )
        chart = px.pie(df, values="Percent", names="Speaker", title="Conversation share by speaker")
        st.plotly_chart(chart, use_container_width=True)

    with right:
        st.subheader("Conversation Summary")
        st.write(summary)
        st.markdown("#### Speaker focus")
        for speaker, focus in speaker_focus.items():
            st.write(f"**{speaker}:** {focus}")

    st.subheader("Transcript (speaker-tagged)")
    for segment in labeled_transcript:
        st.write(f"`{segment.start:06.2f}s - {segment.end:06.2f}s` **{segment.speaker}:** {segment.text}")

    st.success(
        "Prototype note: current version uses mock speaker diarization. Replace `mock_diarization` with pyannote/Deepgram for production realtime speaker identity."
    )
else:
    st.write("Upload an audio file to begin measuring who spoke the most.")
