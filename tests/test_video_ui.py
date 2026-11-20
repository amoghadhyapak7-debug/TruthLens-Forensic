from monitor.results import present_upload_result
from monitor.app import render_metadata


def test_video_metadata_presentation_includes_unavailable_values(monkeypatch):
    import monitor.app as ui
    calls = {}
    class FakeStreamlit:
        def dataframe(self, value, **kwargs):
            calls["rows"] = value
        def expander(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def code(self, value, **kwargs):
            calls["technical"] = value
        def caption(self, value):
            calls["caption"] = value
    monkeypatch.setattr(ui, "st", FakeStreamlit())
    render_metadata({"modality":"video", "duration_seconds":12.0, "width":1920,
        "height":1080, "resolution":"1920x1080", "video_codec":"h264",
        "fps":30.0, "frame_count":None, "bitrate":None,
        "video_stream_count":1, "audio_stream_count":1, "audio_codec":"aac",
        "sample_rate":48000, "metadata_extraction_status":"complete",
        "stream_readable":True, "stream_metadata":["Stream #0:0 Video: h264"]})
    values = {row["Property"]: row["Value"] for row in calls["rows"]}
    assert values["Width"] == 1920 and values["Duration (seconds)"] == 12.0
    assert values["Frame count"] == "Unavailable" and values["Bitrate (bps)"] == "Unavailable"
    assert values["Stream readability"] is True
    assert "Video: h264" in calls["technical"]


def test_upload_result_keeps_validation_and_eligibility_distinct():
    view = present_upload_result({"status":"completed", "original_filename":"over.mp4",
        "analysis_eligibility":{"status":"NOT_READY", "reasons":["Video duration exceeds 10.0 seconds."],"warnings":[]},
        "evidence":{"file_type":"video", "mime_type":"video/mp4", "validation_status":"valid",
            "analysis_eligibility":{"status":"NOT_READY", "reasons":["Video duration exceeds 10.0 seconds."],"warnings":[]},
            "metadata":{"modality":"video", "metadata_extraction_status":"complete", "stream_readable":True}}}, "now")
    assert view["Validation status"] == "valid"
    assert view["Analysis eligibility"]["status"] == "NOT_READY"
    assert view["Metadata"]["stream_readable"] is True


def test_upload_result_does_not_repeat_validation_message_in_generic_error():
    message = "Video contains no readable video stream."
    view = present_upload_result({"status":"completed", "original_filename":"bad.mp4",
        "errors":[message], "analysis_eligibility":{"status":"NOT_READY", "reasons":[message], "warnings":[]},
        "evidence":{"validation_status":"unreadable", "metadata":{"modality":"video"}}}, "now")
    assert view["Analysis eligibility"]["reasons"] == [message]
    assert view["Error"] == "—"
