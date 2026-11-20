import io
import subprocess
import wave
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter

from app.ingest.config import LIMITS, MIB
from app.ingest.metadata import analyzer_limit_reasons
from app.ingest.service import IngestService


def png(size=(8, 6), fmt="PNG", color=(30, 90, 180)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format=fmt)
    return buf.getvalue()


def wav_bytes(path: Path, seconds=0.5, rate=8000):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\0\0" * int(seconds * rate))
    return path.read_bytes()


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_video(path: Path, seconds: float, color="red"):
    result = subprocess.run([ffmpeg(), "-hide_banner", "-f", "lavfi", "-i",
        f"color=c={color}:size=64x48:rate=10:duration={seconds}", "-c:v", "mpeg4",
        "-pix_fmt", "yuv420p", "-y", str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return path.read_bytes()


def make_av_video(path: Path, seconds: float):
    result = subprocess.run([ffmpeg(), "-hide_banner", "-f", "lavfi", "-i",
        f"color=c=red:size=96x64:rate=12:duration={seconds}", "-f", "lavfi", "-i",
        f"sine=frequency=440:duration={seconds}", "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", "-y", str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return path.read_bytes()


def make_encoded_audio(path: Path, codec: str, frequency=440):
    result = subprocess.run([ffmpeg(), "-hide_banner", "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:duration=0.5", "-c:a", codec, "-y", str(path)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return path.read_bytes()


def test_supported_images_and_mismatched_content(tmp_path):
    service = IngestService(tmp_path, limits=replace(LIMITS, image_max_bytes=10**6, image_max_pixels=1000))
    for index, (filename, fmt) in enumerate([("a.png", "PNG"), ("b.jpg", "JPEG"), ("c.jpeg", "JPEG"), ("d.webp", "WEBP")]):
        result = service.ingest(filename, io.BytesIO(png(fmt=fmt, color=(30 + index, 90, 180))))
        assert result["status"] == "completed"
        assert result["evidence"]["file_type"] == "image"
        assert result["metadata"]["pixel_count"] == 48
        assert service.get_analyzer_readiness(result["evidence_id"])["status"] == "READY"
    mismatched_content = png(fmt="PNG", color=(99, 12, 210))
    mismatch = service.ingest("wrong-extension.jpg", io.BytesIO(mismatched_content))
    assert mismatch["status"] == "completed"
    assert mismatch["evidence"]["validation_status"] == "invalid_content"
    assert mismatch["analysis_eligibility"]["status"] == "NOT_READY"
    assert Path(mismatch["evidence"]["file_path"]).read_bytes() == mismatched_content
    assert service.ingest("unsupported.gif", io.BytesIO(b"GIF89a"))["status"] == "failed"


def test_image_corruption_size_and_resolution_eligibility(tmp_path):
    service = IngestService(tmp_path, limits=replace(LIMITS, image_max_bytes=10, image_max_pixels=4))
    oversized = service.ingest("big.png", io.BytesIO(png((3, 3))))
    assert oversized["status"] == "completed"
    assert oversized["analysis_eligibility"]["status"] == "NOT_READY"
    assert any("file-size" in reason for reason in oversized["analysis_eligibility"]["reasons"])
    assert any("resolution" in reason for reason in oversized["analysis_eligibility"]["reasons"])
    corrupt = service.ingest("corrupt.png", io.BytesIO(b"not a png"))
    assert corrupt["status"] == "completed"
    assert corrupt["analysis_eligibility"]["status"] == "NOT_READY"
    assert Path(corrupt["evidence"]["file_path"]).read_bytes() == b"not a png"


def test_video_duration_metadata_boundary_and_missing_tool(tmp_path, monkeypatch):
    service = IngestService(tmp_path / "video", limits=replace(LIMITS, video_max_duration_seconds=1.0))
    under = service.ingest("under.mp4", io.BytesIO(make_video(tmp_path / "under.mp4", 0.5, "red")))
    assert under["analysis_eligibility"]["status"] == "READY"
    assert under["metadata"]["duration_seconds"] <= 1.0
    assert under["metadata"]["width"] == 64 and under["metadata"]["frame_count"] > 0
    assert under["metadata"]["video_stream_count"] == 1
    assert under["metadata"]["stream_readable"] is True
    assert under["metadata"]["metadata_extraction_status"] == "complete"
    boundary = service.ingest("boundary.mp4", io.BytesIO(make_video(tmp_path / "boundary.mp4", 1.0, "green")))
    assert boundary["analysis_eligibility"]["status"] == "READY"
    over = service.ingest("over.mp4", io.BytesIO(make_video(tmp_path / "over.mp4", 1.3, "blue")))
    assert over["analysis_eligibility"]["status"] == "NOT_READY"
    assert any("duration exceeds" in reason for reason in over["analysis_eligibility"]["reasons"])
    assert over["evidence"]["validation_status"] == "valid"
    assert service.get_analyzer_readiness(over["evidence_id"])["reasons"] == over["analysis_eligibility"]["reasons"]
    mov = service.ingest("quicktime.mov", io.BytesIO(make_video(tmp_path / "quicktime.mov", 0.4, "purple")))
    assert mov["status"] == "completed" and mov["metadata"]["duration_seconds"] <= 1.0
    corrupt = service.ingest("corrupt.mp4", io.BytesIO(b"not an mp4 container"))
    assert corrupt["status"] == "completed" and corrupt["analysis_eligibility"]["status"] == "NOT_READY"
    import app.ingest.metadata as metadata_module
    monkeypatch.setattr(metadata_module, "imageio_ffmpeg", None)
    monkeypatch.setattr(metadata_module, "cv2", None)
    missing = service.ingest("tool-missing.mp4", io.BytesIO(make_video(tmp_path / "missing.mp4", 0.5, "yellow")))
    assert missing["status"] == "completed" and missing["analysis_eligibility"]["status"] == "NOT_READY"


def test_video_audio_stream_metadata_audio_only_and_deduplicated_errors(tmp_path):
    service = IngestService(tmp_path / "av")
    av_result = service.ingest("with-audio.mp4", io.BytesIO(make_av_video(tmp_path / "av.mp4", 1.2)))
    metadata = av_result["metadata"]
    assert metadata["video_stream_count"] == 1 and metadata["audio_stream_count"] == 1
    assert metadata["audio_codec"] == "aac" and metadata["sample_rate"] == 44100
    assert metadata["resolution"] == "96x64" and metadata["stream_readable"] is True
    assert av_result["evidence"]["validation_status"] == "valid"
    assert av_result["analysis_eligibility"]["reasons"] == []
    audio_path = tmp_path / "audio-only.mp4"
    data = make_encoded_audio(audio_path, "aac")
    audio_only = service.ingest("audio-only.mp4", io.BytesIO(data))
    assert audio_only["metadata"]["video_stream_count"] == 0
    assert audio_only["metadata"]["audio_stream_count"] == 1
    assert audio_only["evidence"]["validation_status"] == "valid"
    assert audio_only["analysis_eligibility"]["reasons"].count("MP4 contains streams but no video stream was detected.") == 1
    assert service.get_analyzer_readiness(audio_only["evidence_id"])["reasons"] == audio_only["analysis_eligibility"]["reasons"]


def test_duplicate_video_reextracts_attempt_metadata_and_never_assigns_id(tmp_path):
    service = IngestService(tmp_path / "duplicate-video")
    data = make_video(tmp_path / "duplicate.mp4", 0.6, "yellow")
    first = service.ingest("registered.mp4", io.BytesIO(data))
    duplicate = service.ingest("second-attempt.mp4", io.BytesIO(data))
    assert first["evidence_id"] == "EVD-001"
    assert duplicate["status"] == "duplicate"
    assert "evidence_id" not in duplicate
    assert duplicate["duplicate_reference"] == "EVD-001"
    assert duplicate["metadata_source_label"].startswith("This upload")
    assert duplicate["validation_status"] == "valid"
    assert duplicate["metadata"]["metadata_extraction_status"] == "complete"
    assert duplicate["metadata"]["duration_seconds"] <= 10
    assert len(service.list_evidence()) == 1


def test_duplicate_audio_only_mp4_metadata_status_matches_current_attempt(tmp_path):
    path = tmp_path / "audio-only-long.mp4"
    encoded = subprocess.run([ffmpeg(), "-hide_banner", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=10.41", "-c:a", "aac", "-y", str(path)],
        capture_output=True, text=True, timeout=60)
    assert encoded.returncode == 0, encoded.stderr
    service = IngestService(tmp_path / "long-audio-only")
    first = service.ingest("test audio one.mp4", io.BytesIO(path.read_bytes()))
    duplicate = service.ingest("test audio one.mp4", io.BytesIO(path.read_bytes()))
    assert duplicate["status"] == "duplicate" and "evidence_id" not in duplicate
    assert duplicate["duplicate_reference"] == first["evidence_id"]
    assert duplicate["metadata_source_label"].startswith("This upload")
    assert duplicate["validation_status"] == "valid"
    assert duplicate["metadata"]["metadata_extraction_status"] == "partial"
    assert duplicate["metadata"]["container_format"] == "mov"
    assert duplicate["metadata"]["duration_seconds"] >= 10.4
    assert duplicate["metadata"]["video_stream_count"] == 0
    assert duplicate["metadata"]["audio_stream_count"] == 1
    assert duplicate["metadata"]["stream_readable"] is False
    assert len(service.list_evidence()) == 1


def test_audio_wav_mp3_m4a_duration_and_invalid(tmp_path, monkeypatch):
    service = IngestService(tmp_path / "audio", limits=replace(LIMITS, audio_max_duration_seconds=1.0))
    short = service.ingest("short.wav", io.BytesIO(wav_bytes(tmp_path / "short.wav", .5)))
    assert short["analysis_eligibility"]["status"] == "READY"
    assert short["metadata"]["sample_rate"] == 8000
    boundary = service.ingest("boundary.wav", io.BytesIO(wav_bytes(tmp_path / "boundary.wav", 1.0)))
    assert boundary["analysis_eligibility"]["status"] == "READY"
    over = service.ingest("over.wav", io.BytesIO(wav_bytes(tmp_path / "over.wav", 1.25)))
    assert over["analysis_eligibility"]["status"] == "NOT_READY"
    encoded = {}
    for ext, codec in [(".mp3", "libmp3lame"), (".m4a", "aac")]:
        data = make_encoded_audio(tmp_path / f"source{ext}", codec)
        encoded[ext] = data
        result = service.ingest(f"source{ext}", io.BytesIO(data))
        assert result["status"] == "completed" and result["metadata"]["duration_seconds"]
    wrong_audio_extension = service.ingest("wrong.m4a", io.BytesIO(make_encoded_audio(tmp_path / "wrong-source.mp3", "libmp3lame", 550)))
    assert wrong_audio_extension["analysis_eligibility"]["status"] == "NOT_READY"
    invalid = service.ingest("bad.wav", io.BytesIO(b"not wave"))
    assert invalid["analysis_eligibility"]["status"] == "NOT_READY"
    import app.ingest.metadata as metadata_module
    import imageio_ffmpeg
    del imageio_ffmpeg
    monkeypatch.setattr(metadata_module, "MutagenFile", None)
    no_audio_tool = service.ingest("no-audio-tool.mp3", io.BytesIO(make_encoded_audio(tmp_path / "no-tool.mp3", "libmp3lame", 660)))
    assert no_audio_tool["status"] == "completed" and no_audio_tool["analysis_eligibility"]["status"] == "NOT_READY"


def test_text_pdf_docx_csv_and_json_extractors(tmp_path):
    service = IngestService(tmp_path / "formats")
    for filename, content in [("note.txt", b"hello world"), ("note.md", b"# Title\nBody")]:
        result = service.ingest(filename, io.BytesIO(content))
        assert result["metadata"]["format"] == filename.rsplit(".", 1)[1].upper()
        assert result["metadata"]["encoding"] == "utf-8"
        assert result["metadata"]["word_count"] > 0
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter(); writer.add_blank_page(width=100, height=100)
    with pdf_path.open("wb") as stream: writer.write(stream)
    pdf = service.ingest("sample.pdf", pdf_path.open("rb"))
    assert pdf["metadata"]["page_count"] == 1 and pdf["metadata"]["extraction_method"] == "pypdf"
    docx = io.BytesIO()
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>forensic report</w:t></w:r></w:p></w:body></w:document>')
    document = service.ingest("report.docx", io.BytesIO(docx.getvalue()))
    assert document["metadata"]["extracted_text"] == "forensic report"
    csv = service.ingest("table.csv", io.BytesIO(b"name,score\na,1\nb,2"))
    assert csv["metadata"]["rows"] == 2 and csv["metadata"]["columns"] == 2
    js = service.ingest("data.json", io.BytesIO(b'{"a":[1,2]}'))
    assert js["metadata"]["valid"] and js["metadata"]["object_keys"] == ["a"]
    assert js["metadata"]["nesting_depth"] == 2
    assert service.ingest("bad.json", io.BytesIO(b"{"))["analysis_eligibility"]["status"] == "NOT_READY"
    assert service.ingest("bad.csv", io.BytesIO(b"a,b\n1,2,3"))["analysis_eligibility"]["status"] == "NOT_READY"
    assert service.ingest("bad.docx", io.BytesIO(b"not a zip"))["analysis_eligibility"]["status"] == "NOT_READY"
    assert service.ingest("empty.txt", io.BytesIO(b""))["status"] == "failed"
    assert service.ingest("empty.csv", io.BytesIO(b""))["status"] == "failed"
    encrypted_writer = PdfWriter(); encrypted_writer.add_blank_page(width=10, height=10); encrypted_writer.encrypt("secret")
    encrypted_path = tmp_path / "encrypted.pdf"
    with encrypted_path.open("wb") as stream: encrypted_writer.write(stream)
    encrypted = service.ingest("encrypted.pdf", encrypted_path.open("rb"))
    assert encrypted["status"] == "completed" and encrypted["analysis_eligibility"]["status"] == "NOT_READY"
    broken_pdf = service.ingest("broken.pdf", io.BytesIO(b"%PDF-not actually a pdf"))
    assert broken_pdf["analysis_eligibility"]["status"] == "NOT_READY"


def test_document_and_structured_byte_limits_are_eligibility_only(tmp_path):
    limits = replace(LIMITS, document_max_bytes=1, structured_max_bytes=1, audio_max_bytes=1)
    service = IngestService(tmp_path, limits=limits)
    writer = PdfWriter(); writer.add_blank_page(width=10, height=10)
    pdf = io.BytesIO(); writer.write(pdf)
    document = service.ingest("large.pdf", io.BytesIO(pdf.getvalue()))
    assert document["status"] == "completed"
    assert any("file-size" in reason for reason in document["analysis_eligibility"]["reasons"])
    structured = service.ingest("large.json", io.BytesIO(b'{"x":1}'))
    assert structured["status"] == "completed"
    assert any("file-size" in reason for reason in structured["analysis_eligibility"]["reasons"])
    audio = service.ingest("large.wav", io.BytesIO(wav_bytes(tmp_path / "x.wav", .1)))
    assert audio["status"] == "completed"
    assert any("file-size" in reason for reason in audio["analysis_eligibility"]["reasons"])


def test_text_limit_and_analyzer_contract_integrity(tmp_path):
    service = IngestService(tmp_path, limits=replace(LIMITS, text_max_characters=20, structured_max_characters=20))
    data = b"x" * 21
    long_text = service.ingest("long.txt", io.BytesIO(data))
    assert long_text["analysis_eligibility"]["status"] == "NOT_READY"
    assert len(long_text["metadata"]["extracted_text"]) == 20
    assert Path(long_text["evidence"]["file_path"]).read_bytes() == data
    valid = service.ingest("valid.json", io.BytesIO(b'{"ok":true}'))
    contract = service.build_analyzer_input(valid["evidence_id"])
    assert set(contract) == {"evidence_id", "file_path", "original_filename", "file_type", "mime_type", "file_size", "sha256", "ingestion_metadata"}
    assert contract["file_type"] == "structured_data"
    assert Path(contract["file_path"]).stat().st_size == contract["file_size"]
    assert contract["ingestion_metadata"]["valid"] is True
    Path(contract["file_path"]).write_bytes(b"corrupted!!")
    assert service.get_analyzer_readiness(valid["evidence_id"])["status"] == "NOT_READY"
    with pytest.raises(ValueError, match="integrity"):
        service.build_analyzer_input(valid["evidence_id"])
    assert service.get_analyzer_readiness("EVD-404")["status"] == "NOT_READY"
    missing = service.ingest("missing.json", io.BytesIO(b'{"missing":true}'))
    Path(missing["evidence"]["file_path"]).unlink()
    assert service.get_analyzer_readiness(missing["evidence_id"])["status"] == "NOT_READY"
    with pytest.raises(ValueError, match="missing"):
        service.build_analyzer_input(missing["evidence_id"])


def test_exact_video_duration_boundary():
    limits = replace(LIMITS, video_max_duration_seconds=10.0)
    assert analyzer_limit_reasons("video", 100, {"duration_seconds": 10.0}, limits) == []
    assert analyzer_limit_reasons("video", 100, {"duration_seconds": 10.001}, limits)


def test_central_limits_and_eligibility_size_boundaries():
    assert LIMITS.max_files_per_batch == 10 and LIMITS.max_batch_bytes == 200 * MIB
    assert LIMITS.image_max_bytes == 20 * MIB and LIMITS.image_max_pixels == 100_000_000
    assert LIMITS.video_max_bytes == 100 * MIB and LIMITS.video_max_duration_seconds == 10
    assert LIMITS.audio_max_bytes == 25 * MIB and LIMITS.audio_max_duration_seconds == 20
    assert LIMITS.document_max_bytes == 10 * MIB and LIMITS.structured_max_bytes == 10 * MIB
    image_limits = replace(LIMITS, image_max_bytes=20, image_max_pixels=100)
    assert analyzer_limit_reasons("image", 20, {"pixel_count": 100}, image_limits) == []
    assert analyzer_limit_reasons("image", 21, {"pixel_count": 101}, image_limits)
    audio_limits = replace(LIMITS, audio_max_duration_seconds=20)
    assert analyzer_limit_reasons("audio", 1, {"duration_seconds": 20}, audio_limits) == []
    assert analyzer_limit_reasons("audio", 1, {"duration_seconds": 20.01}, audio_limits)


def test_limit_environment_overrides_are_centralized(monkeypatch):
    from app.ingest.config import IngestLimits
    monkeypatch.setenv("TRUTHLENS_LIMIT_VIDEO_MAX_DURATION_SECONDS", "8.5")
    monkeypatch.setenv("TRUTHLENS_LIMIT_MAX_FILES_PER_BATCH", "7")
    configured = IngestLimits.from_env()
    assert configured.video_max_duration_seconds == 8.5
    assert configured.max_files_per_batch == 7


def test_default_text_and_structured_character_caps(tmp_path):
    service = IngestService(tmp_path)
    long_text = service.ingest("long.md", io.BytesIO(b"a" * 100_001))
    assert long_text["status"] == "completed"
    assert long_text["analysis_eligibility"]["status"] == "NOT_READY"
    assert long_text["metadata"]["extracted_character_count"] == 100_001
    long_json_content = ('"' + ('a' * 100_001) + '"').encode()
    long_json = service.ingest("long.json", io.BytesIO(long_json_content))
    assert long_json["status"] == "completed"
    assert long_json["analysis_eligibility"]["status"] == "NOT_READY"


def test_unexpected_metadata_extractor_failure_keeps_original(tmp_path, monkeypatch):
    import app.ingest.service as service_module
    service = IngestService(tmp_path)
    monkeypatch.setattr(service_module, "extract_metadata", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("probe crashed")))
    result = service.ingest("metadata-failure.txt", io.BytesIO(b"preserve these bytes"))
    assert result["status"] == "completed"
    assert result["analysis_eligibility"]["status"] == "NOT_READY"
    assert "probe crashed" in result["warnings"][0]
    assert Path(result["evidence"]["file_path"]).read_bytes() == b"preserve these bytes"
