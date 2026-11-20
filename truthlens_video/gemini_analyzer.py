"""TruthLens Video Forensics - Gemini AI Video Understanding Module

Integrates with the Google Gemini API using the official google-genai SDK.
Provides semantic video understanding, event summarization, and qualitative
visual anomaly observations while maintaining strict forensic integrity and
honesty (clearly separating AI interpretive notes from deterministic measurements).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from truthlens_video.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
)
from truthlens_video.frame_extractor import ExtractedFrame
from truthlens_gemini_auth import DEFAULT_MODEL, get_gemini_client, resolve_api_key

logger = logging.getLogger(__name__)


class GeminiVideoAnalyzer:
    """Provides AI-assisted video understanding and visual observation via Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.model_name = (model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)).strip()
        if api_key:
            self.api_key = api_key.strip()
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize GenAI client with explicit key: {e}")
                self.client = None
        else:
            # Reusable authorized environment authentication
            self.client = get_gemini_client()
            resolved_key, _ = resolve_api_key()
            self.api_key = resolved_key or ""

    def is_available(self) -> bool:
        """Returns True if Gemini API key and client are configured."""
        return bool(self.client and self.api_key)

    def analyze_video_context(
        self,
        frames: List[ExtractedFrame],
        video_metadata: Dict[str, Any],
        max_prompt_frames: int = 12,
    ) -> Dict[str, Any]:
        """Sends representative video keyframes and metadata to Gemini for semantic understanding.

        Gracefully degrades if the API is unavailable, key is missing, or network fails.
        """
        if not self.is_available():
            return {
                "available": False,
                "model": self.model_name,
                "summary": "AI-assisted video understanding unavailable (GEMINI_API_KEY not set). Traditional forensic signal analysis was completed.",
                "events_described": [],
                "visual_observations": [],
                "ai_findings": [],
                "limitations": [
                    "Gemini API key not configured. Video understanding features were bypassed.",
                    "Forensic assessment relied exclusively on deterministic computer vision measurements.",
                ],
            }

        if not frames:
            return {
                "available": True,
                "model": self.model_name,
                "summary": "No frames available for Gemini analysis.",
                "events_described": [],
                "visual_observations": [],
                "ai_findings": [],
                "limitations": ["No frames provided for AI interpretation."],
            }

        try:
            # Subsample frames evenly up to max_prompt_frames
            step = max(1, len(frames) // max_prompt_frames)
            selected_frames = frames[::step][:max_prompt_frames]

            # Prepare images and timestamps
            contents: List[Any] = []
            prompt_intro = (
                "You are an expert AI video forensic assistant working inside the TruthLens digital forensics platform.\n"
                f"Video Metadata: File: {video_metadata.get('file_name')}, "
                f"Duration: {video_metadata.get('duration_seconds')}s, "
                f"Resolution: {video_metadata.get('resolution', {}).get('width')}x{video_metadata.get('resolution', {}).get('height')}, "
                f"FPS: {video_metadata.get('frame_rate')}.\n\n"
                "You are provided with a sequence of keyframes extracted chronologically from this video.\n"
                "Please analyze the sequence and provide your analysis in standard JSON with the following structure:\n"
                "{\n"
                '  "scene_summary": "Concise factual summary of visible actions, setting, and entities",\n'
                '  "key_events": ["Event 1 at timestamp...", "Event 2..."],\n'
                '  "visual_observations": [\n'
                '    {\n'
                '      "timestamp": 1.2,\n'
                '      "frame_name": "filename",\n'
                '      "aspect": "lighting | anatomy | edges | physics | temporal_consistency",\n'
                '      "observation": "Detailed factual description of visual appearance",\n'
                '      "potential_anomaly_detected": true/false,\n'
                '      "anomaly_description": "Description of why this looks unusual or AI-generated if applicable"\n'
                '    }\n'
                '  ],\n'
                '  "ai_synthesis_indicators": [\n'
                '    "List any subtle generative AI signatures if visible, such as texture smoothing, anatomical errors, morphing backgrounds, or text warping"\n'
                '  ],\n'
                '  "interpretive_conclusion": "Brief forensic observation summary (do NOT claim verified proof)"\n'
                "}\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. Base observations strictly on what is visually apparent in these frames.\n"
                "2. Do NOT hallucinate deepfake or manipulation verdicts. If the video looks normal and consistent, say so.\n"
                "3. Distinguish clearly between natural camera artifacts (lens flare, motion blur, normal compression) and potential generative AI/manipulation artifacts.\n"
            )
            contents.append(prompt_intro)

            for sf in selected_frames:
                frame_path = Path(sf.absolute_path)
                if frame_path.exists():
                    img = Image.open(frame_path)
                    # Resize slightly to reduce payload size while preserving fidelity
                    img.thumbnail((720, 720))
                    contents.append(
                        f"\n[Keyframe at timestamp {sf.timestamp_seconds:.2f}s - File: {frame_path.name}]"
                    )
                    contents.append(img)

            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )

            raw_text = response.text or ""
            parsed = self._safe_parse_json(raw_text)

            # Convert Gemini observations to structured TruthLens findings
            ai_findings: List[Dict[str, Any]] = []
            for obs in parsed.get("visual_observations", []):
                if obs.get("potential_anomaly_detected"):
                    ts = obs.get("timestamp")
                    try:
                        ts_val = float(ts) if ts is not None else 0.0
                    except (ValueError, TypeError):
                        ts_val = 0.0

                    ai_findings.append({
                        "indicator_name": f"gemini_observed_{obs.get('aspect', 'visual_anomaly')}",
                        "description": obs.get("anomaly_description") or obs.get("observation", ""),
                        "timestamp": ts_val,
                        "frame_reference": obs.get("frame_name") or "multiple_keyframes",
                        "severity": "low",  # AI observations default to low/medium severity to prevent false positives
                        "method_used": f"google_gemini_vision_{self.model_name}",
                        "supporting_observations": (
                            f"Observation: {obs.get('observation')}. Aspect: {obs.get('aspect')}."
                        ),
                    })

            for gen_ind in parsed.get("ai_synthesis_indicators", []):
                if gen_ind and len(gen_ind.strip()) > 5:
                    ai_findings.append({
                        "indicator_name": "gemini_synthesis_indicator",
                        "description": gen_ind.strip(),
                        "timestamp": None,
                        "frame_reference": None,
                        "severity": "low",
                        "method_used": f"google_gemini_vision_{self.model_name}",
                        "supporting_observations": "Semantic pattern analysis by Gemini vision model.",
                    })

            return {
                "available": True,
                "model": self.model_name,
                "summary": parsed.get("scene_summary", "Summary not generated."),
                "events_described": parsed.get("key_events", []),
                "visual_observations": parsed.get("visual_observations", []),
                "ai_findings": ai_findings,
                "interpretive_conclusion": parsed.get("interpretive_conclusion", ""),
                "limitations": [
                    "Gemini observations are qualitative and AI-assisted. They do not constitute certified mathematical proof of manipulation.",
                    "Observations must be corroborated with forensic signal metrics and cross-video correlation.",
                ],
            }

        except Exception as e:
            logger.error(f"Gemini video analysis failed: {e}", exc_info=True)
            return {
                "available": False,
                "model": self.model_name,
                "error": str(e),
                "summary": f"Gemini API request failed ({type(e).__name__}). Proceeded with traditional forensic signal analysis.",
                "events_described": [],
                "visual_observations": [],
                "ai_findings": [],
                "limitations": [
                    f"Gemini API error encountered: {str(e)}",
                    "Forensic assessment relied exclusively on deterministic computer vision measurements.",
                ],
            }

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, Any]:
        """Safely parses JSON response from LLM, stripping markdown delimiters if present."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Fallback if text is free-form
            return {
                "scene_summary": cleaned[:300] if cleaned else "No summary available.",
                "key_events": [],
                "visual_observations": [],
                "ai_synthesis_indicators": [],
                "interpretive_conclusion": cleaned[:200] if cleaned else "",
            }
