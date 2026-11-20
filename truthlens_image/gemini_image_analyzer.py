"""TruthLens Image Forensics - Gemini AI Vision Analyzer

Provides qualitative semantic visual understanding and artifact observation
using the official Google GenAI SDK with graceful degradation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from truthlens_image.config import GEMINI_API_KEY, GEMINI_MODEL
from truthlens_gemini_auth import DEFAULT_MODEL, get_gemini_client, resolve_api_key

logger = logging.getLogger(__name__)


class GeminiImageAnalyzer:
    """Analyzes image content and qualitative visual integrity via Google Gemini."""

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
                logger.warning(f"Could not initialize Google GenAI client with explicit key: {e}")
                self.client = None
        else:
            self.client = get_gemini_client()
            resolved_key, _ = resolve_api_key()
            self.api_key = resolved_key or ""

    def is_available(self) -> bool:
        return bool(self.client and self.api_key)

    def analyze_image_context(
        self,
        image_path: Path | str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Queries Gemini for semantic visual analysis, describing entities and visual anomalies."""
        if not self.is_available():
            return {
                "available": False,
                "model": self.model_name,
                "summary": "AI image understanding bypassed (GEMINI_API_KEY not configured). Traditional signal analysis active.",
                "visual_observations": [],
                "ai_findings": [],
                "limitations": [
                    "Gemini API key not configured. Visual semantic interpretation was not performed.",
                    "Forensic assessment relied exclusively on deterministic signal measurements.",
                ],
            }

        path = Path(image_path).resolve()
        if not path.exists():
            return {"available": False, "error": "File does not exist"}

        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                img_rgb.thumbnail((1024, 1024))

            prompt = (
                "You are an expert AI forensic assistant working in the TruthLens digital forensics platform.\n"
                f"Image: {metadata.get('file_name')}, Dimensions: {metadata.get('resolution', {}).get('width')}x{metadata.get('resolution', {}).get('height')}.\n\n"
                "Examine this image and return standard JSON:\n"
                "{\n"
                '  "scene_summary": "Concise factual summary of visible entities, scene setting, and visible text",\n'
                '  "visual_observations": [\n'
                '    {\n'
                '      "aspect": "lighting | anatomy | edges | reflections | texture",\n'
                '      "observation": "Factual description of visual feature",\n'
                '      "potential_anomaly_detected": true/false,\n'
                '      "anomaly_description": "Description of why this looks unusual or manipulated if applicable"\n'
                '    }\n'
                '  ],\n'
                '  "ai_synthesis_indicators": ["List any subtle generative AI signatures if visible"],\n'
                '  "interpretive_conclusion": "Brief conclusion (do NOT claim verified proof)"\n'
                "}\n\n"
                "REQUIREMENTS:\n"
                "1. Base observations strictly on visual evidence.\n"
                "2. Do not hallucinate deepfake verdicts. Describe what you see objectively.\n"
            )

            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, img_rgb],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )

            raw_text = response.text or ""
            parsed = self._safe_parse_json(raw_text)

            ai_findings: List[Dict[str, Any]] = []
            for obs in parsed.get("visual_observations", []):
                if obs.get("potential_anomaly_detected"):
                    ai_findings.append({
                        "indicator_name": f"gemini_observed_{obs.get('aspect', 'visual_anomaly')}",
                        "description": obs.get("anomaly_description") or obs.get("observation", ""),
                        "timestamp": None,
                        "frame_reference": path.name,
                        "severity": "low",
                        "method_used": f"google_gemini_vision_{self.model_name}",
                        "supporting_observations": f"Observation: {obs.get('observation')}",
                    })

            for gen_ind in parsed.get("ai_synthesis_indicators", []):
                if gen_ind and len(gen_ind.strip()) > 5:
                    ai_findings.append({
                        "indicator_name": "gemini_synthesis_indicator",
                        "description": gen_ind.strip(),
                        "timestamp": None,
                        "frame_reference": path.name,
                        "severity": "low",
                        "method_used": f"google_gemini_vision_{self.model_name}",
                        "supporting_observations": "Semantic pattern analysis by Gemini vision model.",
                    })

            return {
                "available": True,
                "model": self.model_name,
                "summary": parsed.get("scene_summary", "Summary not generated."),
                "visual_observations": parsed.get("visual_observations", []),
                "ai_findings": ai_findings,
                "interpretive_conclusion": parsed.get("interpretive_conclusion", ""),
                "limitations": [
                    "Gemini observations provide qualitative semantic context; they do not constitute mathematical proof of manipulation.",
                ],
            }

        except Exception as e:
            logger.error(f"Gemini image analysis failed: {e}")
            return {
                "available": False,
                "model": self.model_name,
                "summary": f"Gemini API request failed ({type(e).__name__}). Proceeded with traditional forensic signal analysis.",
                "visual_observations": [],
                "ai_findings": [],
                "limitations": [f"Gemini API error: {str(e)}"],
            }

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, Any]:
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
            return {
                "scene_summary": cleaned[:250] if cleaned else "No summary available.",
                "visual_observations": [],
                "ai_synthesis_indicators": [],
                "interpretive_conclusion": "",
            }
