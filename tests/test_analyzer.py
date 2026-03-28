from utalktoomuch.analyzer import ConversationAnalyzer, SpeakerSegment, TranscriptSegment


def test_assign_speakers_by_overlap():
    analyzer = ConversationAnalyzer()
    transcript = [
        TranscriptSegment(start=0.0, end=2.0, text="Hello team"),
        TranscriptSegment(start=2.0, end=4.0, text="Thanks for joining"),
    ]
    diarization = [
        SpeakerSegment(speaker="Speaker 1", start=0.0, end=2.5),
        SpeakerSegment(speaker="Speaker 2", start=2.5, end=4.0),
    ]

    labeled = analyzer.assign_speakers(transcript, diarization)

    assert labeled[0].speaker == "Speaker 1"
    assert labeled[1].speaker == "Speaker 2"


def test_speaking_share_adds_to_100():
    analyzer = ConversationAnalyzer()
    transcript = [
        TranscriptSegment(start=0.0, end=3.0, text="A", speaker="Speaker 1"),
        TranscriptSegment(start=3.0, end=5.0, text="B", speaker="Speaker 2"),
    ]

    shares = analyzer.speaking_share(transcript)

    assert round(sum(shares.values()), 5) == 100.0
    assert shares["Speaker 1"] == 60.0
    assert shares["Speaker 2"] == 40.0
