"""
Image Validator — validates file integrity, format, magic bytes, dimensions, and color mode.
Features resilient multi-decoder fallback (PIL + OpenCV + Video Keyframe extraction)
to eliminate 'cannot identify image file' errors.
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ImageValidator:
    SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP", "GIF", "VIDEO"}
    MAX_PIXELS = 50_000_000  # 50 MP
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB hard limit

    MAGIC_SIGNATURES = [
        (b"\xFF\xD8", "JPEG", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "PNG", "image/png"),
        (b"GIF87a", "GIF", "image/gif"),
        (b"GIF89a", "GIF", "image/gif"),
        (b"BM", "BMP", "image/bmp"),
        (b"II*\x00", "TIFF", "image/tiff"),
        (b"MM\x00*", "TIFF", "image/tiff"),
    ]

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}
    DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}

    def validate(
        self,
        image_path: str,
        expected_size: Optional[int] = None,
        expected_mime: Optional[str] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "valid": False,
            "format": None,
            "detected_mime_type": None,
            "magic_bytes_verified": False,
            "mode": None,
            "dimensions": None,
            "file_size_bytes": None,
            "size_match": True,
            "integrity": "unknown",
            "reason": None,
            "is_video": False,
            "is_document": False,
            "keyframe_path": None,
            "normalized_path": None,
        }

        # 1. Existence check
        if not os.path.exists(image_path):
            result["reason"] = f"Evidence file not found: {image_path}"
            return result

        result["file_size_bytes"] = os.path.getsize(image_path)
        if result["file_size_bytes"] == 0:
            result["reason"] = "Uploaded evidence file is empty (0 bytes). Please upload a valid image or video."
            return result

        # Hard 10 MB file-size limit
        if result["file_size_bytes"] > self.MAX_FILE_SIZE:
            mb = result["file_size_bytes"] / (1024 * 1024)
            result["reason"] = (
                f"File size {mb:.2f} MB exceeds the 10 MB limit. "
                "Please upload an evidence file smaller than 10 MB."
            )
            return result

        if expected_size is not None and expected_size > 0:
            if result["file_size_bytes"] != expected_size:
                result["size_match"] = False
                result["size_difference"] = result["file_size_bytes"] - expected_size

        # 2. Inspect Magic Bytes and Extension
        ext = os.path.splitext(image_path)[1].lower()
        header = b""
        try:
            with open(image_path, "rb") as f:
                header = f.read(64)
        except Exception as exc:
            result["reason"] = f"Failed to read file bytes: {exc}"
            return result

        # Check for Document/Text files uploaded by mistake
        if ext in self.DOCUMENT_EXTENSIONS or header.startswith(b"%PDF") or header.startswith(b"PK\x03\x04"):
            result["is_document"] = True
            result["reason"] = (
                f"Document file detected ('{os.path.basename(image_path)}'). "
                "Please use the Text Forensic Analyzer at /text to inspect text and document evidence."
            )
            return result

        # Check for Video files (e.g. MP4, AVI, MOV, WEBM)
        is_video_header = (
            b"ftyp" in header[:16] or
            (header.startswith(b"RIFF") and b"AVI " in header[:16]) or
            header.startswith(b"\x1a\x45\xdf\xa3") or
            ext in self.VIDEO_EXTENSIONS
        )

        if is_video_header:
            return self._handle_video_file(image_path, result)

        # Detect image format from magic bytes
        detected_fmt = None
        detected_mime = None
        for sig, fmt, mime in self.MAGIC_SIGNATURES:
            if header.startswith(sig):
                detected_fmt = fmt
                detected_mime = mime
                break

        # Check for WEBP: RIFF....WEBP
        if not detected_fmt and header.startswith(b"RIFF") and b"WEBP" in header[:16]:
            detected_fmt = "WEBP"
            detected_mime = "image/webp"

        if detected_fmt:
            result["format"] = detected_fmt
            result["detected_mime_type"] = detected_mime
            result["magic_bytes_verified"] = True
        else:
            result["detected_mime_type"] = expected_mime or "application/octet-stream"

        # 3. Multi-Decoder Image Verification (PIL + OpenCV Self-Healing)
        img = None
        pil_format = detected_fmt or "UNKNOWN"

        # Step 3a: Try standard Pillow open
        try:
            from PIL import Image
            test_img = Image.open(image_path)
            test_img.load()
            pil_format = test_img.format or detected_fmt or "JPEG"
            img = test_img
            result["integrity"] = "ok"
        except Exception as pil_err:
            logger.info(f"Pillow open failed ({pil_err}), attempting OpenCV self-healing decoder...")

            # Step 3b: Fallback to OpenCV decoder
            try:
                import cv2
                cv_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if cv_img is not None and cv_img.size > 0:
                    # OpenCV decoded the image successfully!
                    # Normalize and save as clean PNG alongside original
                    normalized_path = image_path + ".normalized.png"
                    cv2.imwrite(normalized_path, cv_img)

                    from PIL import Image
                    img = Image.open(normalized_path)
                    img.load()
                    pil_format = "PNG"
                    result["integrity"] = "ok"
                    result["format"] = "PNG"
                    result["normalized_path"] = normalized_path
                    result["detected_mime_type"] = "image/png"
            except Exception as cv_err:
                logger.warning(f"OpenCV decoder also failed: {cv_err}")

        # If both Pillow and OpenCV failed, check if it's a video file or give a clean, friendly error
        if img is None:
            # Last-ditch check: try VideoCapture
            try:
                import cv2
                cap = cv2.VideoCapture(image_path)
                if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    cap.release()
                    return self._handle_video_file(image_path, result)
            except Exception:
                pass

            result["integrity"] = "corrupted"
            result["reason"] = (
                "Unable to identify or decode image format. Please ensure the file is an intact, "
                "supported image (JPEG, PNG, WEBP, BMP, TIFF) or video (MP4, MOV, AVI)."
            )
            return result

        # 4. Inspect Image Dimensions and Properties
        try:
            result["format"] = pil_format
            result["mode"] = img.mode
            result["dimensions"] = {"width": img.width, "height": img.height}

            if pil_format not in self.SUPPORTED_FORMATS:
                result["reason"] = f"Unsupported image format: {pil_format}. Supported: {sorted(list(self.SUPPORTED_FORMATS))}"
                return result

            pixels = img.width * img.height
            if pixels > self.MAX_PIXELS:
                result["reason"] = f"Image dimensions too large: {pixels} pixels (max {self.MAX_PIXELS})"
                return result

            result["valid"] = True
            return result
        except Exception as exc:
            result["reason"] = f"Failed to inspect image properties: {exc}"
            return result

    def _handle_video_file(self, video_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts representative forensic keyframe from video evidence
        allowing unified forensic verification on video files without crashes.
        """
        try:
            import cv2
            from PIL import Image

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result["reason"] = "Could not open video stream or codec is unsupported."
                return result

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0

            # Seek to 1/3 of the video for a representative forensic keyframe
            target_frame = max(0, frame_count // 3)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if not ret or frame is None:
                # Try first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            cap.release()

            if not ret or frame is None:
                result["reason"] = "Video stream contains zero readable frames."
                return result

            # Save extracted keyframe as JPEG
            keyframe_path = video_path + ".keyframe.jpg"
            cv2.imwrite(keyframe_path, frame)

            # Inspect extracted keyframe
            with Image.open(keyframe_path) as kf:
                result["valid"] = True
                result["format"] = "VIDEO"
                result["detected_mime_type"] = "video/mp4"
                result["is_video"] = True
                result["keyframe_path"] = keyframe_path
                result["mode"] = kf.mode
                result["dimensions"] = {"width": width, "height": height}
                result["video_metadata"] = {
                    "total_frames": frame_count,
                    "fps": round(fps, 2),
                    "duration_seconds": duration_sec,
                    "resolution": f"{width}x{height}",
                    "analyzed_frame_index": target_frame,
                    "keyframe_path": keyframe_path
                }
                result["integrity"] = "ok"
                return result

        except Exception as exc:
            logger.error(f"Video keyframe extraction failed: {exc}")
            result["reason"] = f"Failed to extract video keyframe: {exc}"
            return result

