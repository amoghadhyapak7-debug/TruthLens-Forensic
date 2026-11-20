"""Best-effort, modality-aware metadata extraction from preserved originals."""

from __future__ import annotations

import csv
import codecs
import io
import json
import re
import subprocess
import wave
import warnings
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from app.ingest.config import (
    AUDIO_EXTENSIONS, DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS, LIMITS,
    STRUCTURED_EXTENSIONS, TEXT_EXTENSIONS, VIDEO_EXTENSIONS,
    modality_for_extension,
)

try:
    from PIL import ExifTags, Image
except ImportError:
    Image = None
    ExifTags = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

try:
    from mutagen import File as MutagenFile
except ImportError:
    MutagenFile = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def _decode_text(data: bytes) -> tuple[str | None, str | None]:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            detected = "utf-8-sig" if encoding == "utf-8-sig" and data.startswith(b"\xef\xbb\xbf") else ("utf-8" if encoding == "utf-8-sig" else encoding)
            return data.decode(encoding), detected
        except (UnicodeDecodeError, LookupError):
            continue
    return None, None


def _nesting_depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + max((_nesting_depth(v) for v in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_nesting_depth(v) for v in value), default=0)
    return 0


def _image(path: Path, ext: str, size: int) -> tuple[dict, list[str], list[str], str]:
    meta = {"file_size": size, "format": None, "width": None, "height": None,
            "pixel_count": None, "color_mode": None, "exif": {}, "camera": None,
            "orientation": None}
    warnings_out: list[str] = []
    errors: list[str] = []
    actual_format = None
    if Image is None:
        warnings_out.append("Pillow is unavailable; image content and dimensions could not be verified.")
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with Image.open(path) as image:
                    actual_format = image.format
                    meta.update(format=actual_format, width=image.width, height=image.height,
                                pixel_count=image.width * image.height, color_mode=image.mode)
                    exif = image.getexif()
                    named_exif = {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in exif.items()} if ExifTags else {str(k): str(v) for k, v in exif.items()}
                    meta["exif"] = named_exif
                    meta["orientation"] = named_exif.get("Orientation")
                    meta["camera"] = {key: named_exif[key] for key in ("Make", "Model") if key in named_exif} or None
                with Image.open(path) as verifier:
                    verifier.verify()
            expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}[ext]
            if actual_format != expected:
                errors.append(f"File extension {ext} does not match detected image format {actual_format}.")
        except Exception as exc:
            errors.append(f"Image is corrupted or unreadable: {exc}")
    return meta, warnings_out, errors, actual_format or ""


def _ffmpeg_probe(path: Path) -> tuple[dict, list[str], list[str]]:
    meta = {"duration_seconds": None, "width": None, "height": None, "fps": None,
            "frame_count": None, "video_codec": None, "container_format": None,
            "resolution": None, "bitrate": None, "video_stream_count": 0,
            "audio_stream_count": 0, "audio_streams_present": False,
            "audio_codec": None, "sample_rate": None, "metadata_extraction_status": "unavailable",
            "stream_readable": False, "stream_metadata": [], "technical_details": []}
    warnings_out: list[str] = []
    errors: list[str] = []
    if imageio_ffmpeg is None:
        warnings_out.append("FFmpeg metadata tool is unavailable; video duration and streams could not be verified.")
        meta["metadata_extraction_status"] = "tool_unavailable"
    else:
        try:
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            text = completed.stderr + "\n" + completed.stdout
            meta["technical_details"] = [line.strip() for line in text.splitlines() if line.strip()][-20:]
            container = re.search(r"Input #\d+,\s*([^,\s]+)", text)
            if container:
                meta["container_format"] = container.group(1)
            duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
            if duration:
                meta["duration_seconds"] = int(duration.group(1)) * 3600 + int(duration.group(2)) * 60 + float(duration.group(3))
            streams = [line.strip() for line in text.splitlines()
                       if re.search(r"Stream #\d+:\d+", line) and " -> " not in line
                       and not line.lstrip().startswith("Output #")]
            streams = [line for line in streams if "Stream mapping" not in line]
            # The one-frame decode adds an output video stream (wrapped_avframe);
            # it is not part of the input container's stream inventory.
            streams = [line for line in streams if "wrapped_avframe" not in line]
            meta["stream_metadata"] = streams
            video_streams = [line for line in streams if "Video:" in line]
            audio_streams = [line for line in streams if "Audio:" in line]
            meta["video_stream_count"] = len(video_streams)
            meta["audio_stream_count"] = len(audio_streams)
            meta["audio_streams_present"] = bool(audio_streams)
            audio_line = audio_streams[0] if audio_streams else None
            if audio_line:
                audio_codec = re.search(r"Audio:\s*([^,\s]+)", audio_line)
                sample_rate = re.search(r"(\d+)\s*Hz", audio_line)
                if audio_codec:
                    meta["audio_codec"] = audio_codec.group(1)
                if sample_rate:
                    meta["sample_rate"] = int(sample_rate.group(1))
            video_line = next((line for line in streams if "Video:" in line), None)
            # FFmpeg's diagnostic stream listing is present even when it cannot
            # decode a frame. A one-frame mapped decode verifies readability.
            meta["stream_readable"] = bool(video_line and completed.returncode == 0)
            if not video_line:
                has_non_video_stream = bool(streams)
                message = ("MP4 contains streams but no video stream was detected."
                           if has_non_video_stream else "Video contains no readable video stream.")
                errors.append(message)
            if video_line:
                codec = re.search(r"Video:\s*([^,\s]+)", video_line)
                size = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
                fps = re.search(r"([\d.]+)\s*fps", video_line)
                if codec:
                    meta["video_codec"] = codec.group(1)
                if size:
                    meta["width"], meta["height"] = map(int, size.groups())
                    meta["resolution"] = f"{meta['width']}x{meta['height']}"
                if fps:
                    meta["fps"] = float(fps.group(1))
            bitrate = re.search(r"bitrate:\s*(\d+)\s*kb/s", text, re.I)
            if bitrate:
                meta["bitrate"] = int(bitrate.group(1)) * 1000
            if video_line and completed.returncode != 0:
                errors.append("Video stream was detected but FFmpeg could not decode a frame.")
            if meta["duration_seconds"] is None:
                warnings_out.append("Video metadata tool could not determine duration.")
            meta["metadata_extraction_status"] = "complete" if meta["stream_readable"] else ("partial" if streams else "failed")
        except subprocess.TimeoutExpired:
            warnings_out.append("FFmpeg metadata extraction timed out.")
            meta["metadata_extraction_status"] = "timeout"
        except Exception as exc:
            warnings_out.append(f"FFmpeg metadata extraction failed: {exc}")
            meta["metadata_extraction_status"] = "failed"
    if cv2 is not None:
        capture = cv2.VideoCapture(str(path))
        try:
            if capture.isOpened():
                count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = capture.get(cv2.CAP_PROP_FPS)
                width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if count > 0:
                    meta["frame_count"] = int(count)
                if fps > 0:
                    meta["fps"] = meta["fps"] or float(fps)
                if width > 0 and height > 0:
                    meta["width"] = meta["width"] or int(width)
                    meta["height"] = meta["height"] or int(height)
                if meta["duration_seconds"] is None and count > 0 and fps > 0:
                    meta["duration_seconds"] = float(count / fps)
                if meta["frame_count"] is not None and meta["fps"] and meta["duration_seconds"] is None:
                    meta["duration_seconds"] = float(meta["frame_count"] / meta["fps"])
        finally:
            capture.release()
    if meta["duration_seconds"] is None:
        warnings_out.append("Video duration is unavailable; analyzer eligibility cannot be confirmed.")
    return meta, warnings_out, errors


def _audio(path: Path, ext: str, size: int) -> tuple[dict, list[str], list[str]]:
    meta = {"duration_seconds": None, "codec": None, "sample_rate": None,
            "channels": None, "bitrate": None, "audio_format": ext[1:], "file_size": size}
    warnings_out: list[str] = []
    errors: list[str] = []
    if ext == ".wav":
        try:
            with wave.open(str(path), "rb") as audio:
                meta.update(duration_seconds=audio.getnframes() / audio.getframerate(),
                            sample_rate=audio.getframerate(), channels=audio.getnchannels(),
                            codec="PCM", bitrate=audio.getframerate() * audio.getnchannels() * audio.getsampwidth() * 8)
        except Exception as exc:
            errors.append(f"WAV is corrupted or unreadable: {exc}")
    elif MutagenFile is None:
        warnings_out.append("Mutagen is unavailable; audio metadata and duration could not be verified.")
    else:
        try:
            audio = MutagenFile(str(path))
            if audio is None or getattr(audio, "info", None) is None:
                errors.append("Audio file is corrupted or its format is not recognized.")
            else:
                info = audio.info
                actual_codec = type(audio).__name__
                if (ext == ".mp3" and actual_codec.lower() != "mp3") or (ext == ".m4a" and actual_codec.lower() not in {"mp4", "m4a"}):
                    errors.append(f"Audio extension {ext} does not match detected codec/container {actual_codec}.")
                meta.update(duration_seconds=getattr(info, "length", None),
                            sample_rate=getattr(info, "sample_rate", None),
                            channels=getattr(info, "channels", None),
                            bitrate=getattr(info, "bitrate", None), codec=actual_codec)
        except Exception as exc:
            errors.append(f"Audio is corrupted or unreadable: {exc}")
    if meta["duration_seconds"] is None and not warnings_out:
        warnings_out.append("Audio duration is unavailable; analyzer eligibility cannot be confirmed.")
    return meta, warnings_out, errors


def _document(path: Path, ext: str, limits=LIMITS, size: int | None = None) -> tuple[dict, list[str], list[str]]:
    meta: dict[str, Any] = {"format": ext[1:].upper(), "extracted_character_count": 0,
                            "word_count": 0, "page_count": None, "title": None, "author": None,
                            "extraction_method": None, "extracted_text": ""}
    warnings_out: list[str] = []
    errors: list[str] = []
    if size is not None and size > limits.document_max_bytes:
        warnings_out.append("Document text extraction skipped because the file exceeds the analyzer file-size limit.")
        return meta, warnings_out, errors
    try:
        if ext == ".pdf":
            if PdfReader is None:
                warnings_out.append("pypdf is unavailable; PDF text and properties could not be extracted.")
                return meta, warnings_out, errors
            reader = PdfReader(str(path), strict=False)
            if reader.is_encrypted:
                errors.append("PDF is encrypted and cannot be read without a password.")
                return meta, warnings_out, errors
            meta["page_count"] = len(reader.pages)
            if meta["page_count"] == 0:
                errors.append("PDF contains no pages.")
                return meta, warnings_out, errors
            info = reader.metadata
            if info:
                meta["title"] = info.title
                meta["author"] = info.author
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            meta["extraction_method"] = "pypdf"
            if not text.strip():
                warnings_out.append("PDF contains no extractable text (it may contain scanned images only).")
        else:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if "word/document.xml" not in names:
                    errors.append("DOCX package is missing word/document.xml.")
                    return meta, warnings_out, errors
                root = ElementTree.fromstring(archive.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                text = "\n".join("".join(node.itertext()) for node in root.findall(".//w:p", ns))
                if "docProps/core.xml" in names:
                    props = ElementTree.fromstring(archive.read("docProps/core.xml"))
                    core_ns = {"dc": "http://purl.org/dc/elements/1.1/", "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"}
                    title = props.find("dc:title", core_ns)
                    author = props.find("dc:creator", core_ns)
                    meta["title"] = title.text if title is not None else None
                    meta["author"] = author.text if author is not None else None
                meta["extraction_method"] = "docx-xml"
    except Exception as exc:
        errors.append(f"Document is encrypted, corrupted, or unreadable: {exc}")
        return meta, warnings_out, errors
    meta["extracted_character_count"] = len(text)
    meta["word_count"] = len(text.split())
    meta["extracted_text"] = text[:limits.text_max_characters]
    if len(text) > limits.text_max_characters:
        warnings_out.append(f"Extracted text exceeds {limits.text_max_characters} characters; preview truncated for intake.")
    if not text.strip():
        warnings_out.append("Document text is empty or unavailable.")
    return meta, warnings_out, errors


def _text(path: Path, ext: str, limits=LIMITS) -> tuple[dict, list[str], list[str]]:
    count = 0
    word_count = 0
    ends_in_word = False
    preview = ""
    encoding = None
    contains_nul = False
    with path.open("rb") as source:
        bom = source.read(3) == b"\xef\xbb\xbf"
    for candidate in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            decoder = codecs.getincrementaldecoder(candidate)(errors="strict")
            count = 0
            word_count = 0
            ends_in_word = False
            preview = ""
            contains_nul = False
            with path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    piece = decoder.decode(chunk)
                    count += len(piece)
                    contains_nul = contains_nul or "\x00" in piece
                    if piece:
                        words = len(piece.split())
                        if ends_in_word and not piece[0].isspace():
                            words -= 1
                        word_count += words
                        ends_in_word = not piece[-1].isspace()
                    if len(preview) < limits.text_max_characters:
                        preview += piece[:limits.text_max_characters - len(preview)]
                tail = decoder.decode(b"", final=True)
                count += len(tail)
                contains_nul = contains_nul or "\x00" in tail
                if tail:
                    words = len(tail.split())
                    if ends_in_word and not tail[0].isspace():
                        words -= 1
                    word_count += words
                if len(preview) < limits.text_max_characters:
                    preview += tail[:limits.text_max_characters - len(preview)]
            encoding = "utf-8-sig" if candidate == "utf-8-sig" and bom else ("utf-8" if candidate == "utf-8-sig" else candidate)
            break
        except (UnicodeDecodeError, LookupError):
            encoding = None
    meta = {"format": ext[1:].upper(), "encoding": encoding,
            "extracted_character_count": count if encoding is not None else None,
            "word_count": word_count if encoding is not None else None,
            "extraction_method": "streaming-python-codecs", "extracted_text": preview if encoding is not None else None}
    warnings_out: list[str] = []
    errors: list[str] = []
    if encoding is None:
        errors.append("Text file encoding could not be decoded.")
    elif contains_nul:
        errors.append("Text file contains NUL bytes and does not appear to be readable text.")
    elif count > limits.text_max_characters:
        warnings_out.append(f"Extracted text exceeds {limits.text_max_characters} characters; preview truncated for intake.")
    return meta, warnings_out, errors


def _structured(path: Path, ext: str, limits=LIMITS, size: int | None = None) -> tuple[dict, list[str], list[str]]:
    if size is not None and size > limits.structured_max_bytes:
        return ({"format": ext[1:].upper(), "encoding": None, "content_character_count": None,
                 "valid": None, "root_type": None, "object_keys": None, "array_count": None,
                 "nesting_depth": None, "rows": None, "columns": None, "headers": None,
                 "delimiter": None},
                ["Structured content validation skipped because the file exceeds the analyzer file-size limit."],
                [])
    data = path.read_bytes()
    text, encoding = _decode_text(data)
    meta: dict[str, Any] = {"format": ext[1:].upper(), "encoding": encoding,
                            "content_character_count": len(text) if text is not None else None,
                            "valid": False, "root_type": None, "object_keys": None,
                            "array_count": None, "nesting_depth": None,
                            "rows": None, "columns": None, "headers": None,
                            "delimiter": None}
    warnings_out: list[str] = []
    errors: list[str] = []
    if text is None:
        errors.append("Structured file encoding could not be decoded.")
        return meta, warnings_out, errors
    if "\x00" in text:
        errors.append("Structured file contains NUL bytes and does not appear to be readable text.")
        return meta, warnings_out, errors
    if ext == ".json":
        try:
            value = json.loads(text)
            meta.update(valid=True, root_type=type(value).__name__, nesting_depth=_nesting_depth(value),
                        object_keys=list(value.keys())[:100] if isinstance(value, dict) else None,
                        array_count=len(value) if isinstance(value, list) else None)
        except json.JSONDecodeError as exc:
            errors.append(f"Malformed JSON: {exc}")
    else:
        try:
            sample = text[:8192]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
            widths = {len(row) for row in rows}
            meta.update(valid=bool(rows), rows=max(0, len(rows) - 1),
                        columns=len(rows[0]) if rows else 0,
                        headers=rows[0] if rows else [], delimiter=delimiter)
            if not rows:
                errors.append("CSV contains no rows or headers.")
            if len(widths) > 1:
                errors.append("CSV rows have inconsistent column counts.")
        except csv.Error as exc:
            errors.append(f"Malformed CSV: {exc}")
    if len(text) > limits.structured_max_characters:
        warnings_out.append(f"Structured content exceeds {limits.structured_max_characters} characters; retained, but not eligible for analysis.")
    return meta, warnings_out, errors


def extract_metadata(path: Path, ext: str, size: int, limits=LIMITS) -> dict[str, Any]:
    modality = modality_for_extension(ext)
    if modality == "image":
        metadata, warnings_out, errors, actual = _image(path, ext, size)
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[ext]
        content_valid = None if Image is None else bool(actual) and not errors
    elif modality == "video":
        metadata, warnings_out, errors = _ffmpeg_probe(path)
        mime = "video/mp4" if ext == ".mp4" else "video/quicktime"
        brand = _ftyp_brand(path)
        content_valid = brand is not None and (ext != ".mov" or brand == "qt  ") and (ext != ".mp4" or brand != "qt  ")
        if not content_valid:
            errors.append(f"Video content brand {brand!r} does not match the declared {ext} format.")
    elif modality == "audio":
        metadata, warnings_out, errors = _audio(path, ext, size)
        mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}[ext]
        content_valid = None if ext != ".wav" and MutagenFile is None else not errors
    elif modality == "document":
        metadata, warnings_out, errors = _document(path, ext, limits, size)
        mime = "application/pdf" if ext == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        content_valid = _magic_matches(path, ext)
        if not content_valid:
            errors.append(f"File content does not match {ext} document format.")
    elif modality == "text":
        metadata, warnings_out, errors = _text(path, ext, limits)
        mime = "text/plain" if ext == ".txt" else "text/markdown"
        content_valid = not errors
    else:
        metadata, warnings_out, errors = _structured(path, ext, limits, size)
        mime = "text/csv" if ext == ".csv" else "application/json"
        content_valid = not errors and metadata.get("valid") is not None
    return {"modality": modality, "mime_type": mime, "metadata": metadata,
            "warnings": warnings_out, "errors": errors, "content_valid": content_valid}


def _ftyp_brand(path: Path) -> str | None:
    try:
        with path.open("rb") as src:
            head = src.read(12)
        if len(head) >= 12 and head[4:8] == b"ftyp":
            return head[8:12].decode("ascii", errors="replace")
        return None
    except OSError:
        return None


def _magic_matches(path: Path, ext: str) -> bool:
    try:
        with path.open("rb") as src:
            head = src.read(8)
        if ext == ".pdf":
            return head.startswith(b"%PDF-")
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            return {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}.issubset(names)
    except Exception:
        return False


def analyzer_limit_reasons(modality: str, size: int, metadata: dict, limits=LIMITS) -> list[str]:
    reasons: list[str] = []
    if modality == "image":
        if size > limits.image_max_bytes:
            reasons.append(f"Image exceeds the {limits.image_max_bytes}-byte analyzer file-size limit.")
        pixels = metadata.get("pixel_count")
        if pixels is not None and pixels > limits.image_max_pixels:
            reasons.append(f"Image resolution exceeds {limits.image_max_pixels} pixels.")
    elif modality == "video":
        if size > limits.video_max_bytes:
            reasons.append(f"Video exceeds the {limits.video_max_bytes}-byte analyzer file-size limit.")
        duration = metadata.get("duration_seconds")
        if duration is None:
            reasons.append("Video duration could not be verified.")
        elif duration > limits.video_max_duration_seconds:
            reasons.append(f"Video duration exceeds {limits.video_max_duration_seconds} seconds.")
    elif modality == "audio":
        if size > limits.audio_max_bytes:
            reasons.append(f"Audio exceeds the {limits.audio_max_bytes}-byte analyzer file-size limit.")
        duration = metadata.get("duration_seconds")
        if duration is None:
            reasons.append("Audio duration could not be verified.")
        elif duration > limits.audio_max_duration_seconds:
            reasons.append(f"Audio duration exceeds {limits.audio_max_duration_seconds} seconds.")
    elif modality == "document":
        if size > limits.document_max_bytes:
            reasons.append(f"Document exceeds the {limits.document_max_bytes}-byte analyzer file-size limit.")
        if metadata.get("extracted_character_count", 0) > limits.text_max_characters:
            reasons.append(f"Extracted document text exceeds {limits.text_max_characters} characters.")
        if metadata.get("extracted_character_count") == 0 and metadata.get("extraction_method"):
            reasons.append("Document contains no extractable text.")
    elif modality == "text":
        if metadata.get("extracted_character_count") is None:
            reasons.append("Text could not be decoded.")
        elif metadata["extracted_character_count"] > limits.text_max_characters:
            reasons.append(f"Text exceeds {limits.text_max_characters} characters.")
    elif modality == "structured_data":
        if size > limits.structured_max_bytes:
            reasons.append(f"Structured file exceeds the {limits.structured_max_bytes}-byte analyzer file-size limit.")
        if (metadata.get("content_character_count") or 0) > limits.structured_max_characters:
            reasons.append(f"Structured content exceeds {limits.structured_max_characters} characters.")
    return reasons
