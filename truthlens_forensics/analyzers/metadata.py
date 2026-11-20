"""
Metadata, EXIF, IPTC, and Header Consistency Analyzer
Extracts embedded hardware, camera, software, timestamp, and GPS metadata.
Checks for editing software signatures, timestamp contradictions, and stripped headers.
"""
import os
import traceback
from typing import Dict, Any, List


class MetadataAnalyzer:
    EDITING_SOFTWARE_KEYWORDS = [
        "photoshop", "gimp", "lightroom", "affinity", "paint", "canva",
        "snapseed", "facetune", "vsco", "picsart", "midjourney", "stable diffusion",
        "dall-e", "firefly", "runway", "topaz", "capture one", "krea", "comfyui"
    ]

    def analyze(self, image_path: str, evidence_id: str = "EVD-001") -> Dict[str, Any]:
        result = {
            "engine": "metadata_analysis",
            "camera_present": False,
            "camera_info": {"make": None, "model": None},
            "editing_software_detected": False,
            "software_tag": None,
            "timestamp_consistent": True,
            "timestamps": {"created": None, "modified": None},
            "stripped_metadata": False,
            "gps_present": False,
            "exif_raw": {},
            "flags": [],
            "interpretation": "",
            "artifacts": [],
            "limitations": [
                "EXIF metadata is stored in plain text within image file headers and can be modified or forged.",
                "Social media platforms (Twitter/X, WhatsApp, Instagram) routinely strip all EXIF metadata for privacy; absence of EXIF alone does not prove malicious manipulation.",
                "Timezone discrepancies between camera internal clocks and computer systems can create apparent timestamp mismatches."
            ],
        }

        try:
            import exifread
            from PIL import Image
            from PIL.ExifTags import TAGS

            pil_exif = {}
            try:
                with Image.open(image_path) as img:
                    raw_exif = img._getexif() if hasattr(img, "_getexif") else None
                    if raw_exif:
                        for tag_id, value in raw_exif.items():
                            tag = TAGS.get(tag_id, str(tag_id))
                            try:
                                pil_exif[tag] = str(value)[:250]
                            except Exception:
                                pil_exif[tag] = "<unparseable>"
            except Exception:
                pass

            exifread_data = {}
            try:
                with open(image_path, "rb") as f:
                    tags = exifread.process_file(f, stop_tag="UNDEF", details=False)
                for k, v in tags.items():
                    try:
                        exifread_data[k] = str(v)[:250]
                    except Exception:
                        exifread_data[k] = "<unparseable>"
            except Exception:
                pass

            merged = {**exifread_data, **pil_exif}
            result["exif_raw"] = merged

            # 1. Camera Presence
            make = merged.get("Image Make", merged.get("Make"))
            model = merged.get("Image Model", merged.get("Model"))
            if make or model:
                result["camera_present"] = True
                result["camera_info"] = {"make": make, "model": model}

            # 2. Editing Software Signature
            software = merged.get("Image Software", merged.get("Software"))
            if software:
                result["software_tag"] = software
                low_sw = software.lower()
                if any(kw in low_sw for kw in self.EDITING_SOFTWARE_KEYWORDS):
                    result["editing_software_detected"] = True
                    result["flags"].append(f"Editing software detected in EXIF: '{software}'")

            # 3. Timestamp Consistency
            created = merged.get("EXIF DateTimeOriginal", merged.get("DateTimeOriginal"))
            modified = merged.get("Image DateTime", merged.get("DateTime"))
            result["timestamps"] = {"created": created, "modified": modified}

            if created and modified and created != modified:
                result["timestamp_consistent"] = False
                result["flags"].append(f"Timestamp mismatch: creation ({created}) vs modification ({modified})")

            # 4. GPS Metadata
            has_gps = any("GPS" in k for k in merged)
            result["gps_present"] = has_gps

            # 5. Stripped Metadata check
            if not merged or (not make and not model and not created):
                result["stripped_metadata"] = True
                result["flags"].append("Camera EXIF metadata is completely absent/stripped")

            # Synthesis
            if result["editing_software_detected"]:
                result["interpretation"] = (
                    f"Metadata confirms post-processing by digital editing software ('{software}'). "
                    + (f"Timestamp divergence noted." if not result["timestamp_consistent"] else "")
                )
            elif result["camera_present"]:
                result["interpretation"] = (
                    f"Metadata contains authentic camera hardware signatures ({make or ''} {model or ''}). "
                    "Consistent with direct photographic acquisition."
                )
            elif result["stripped_metadata"]:
                result["interpretation"] = (
                    "Image headers contain no camera or exposure metadata. Typical of AI-generated content, "
                    "messaging platform stripping, or deliberate scrubbing."
                )
            else:
                result["interpretation"] = "Metadata is present with standard headers."

        except Exception as exc:
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()

        return result

