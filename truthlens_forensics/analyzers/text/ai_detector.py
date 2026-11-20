from __future__ import annotations
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AITextDetector:
    """
    Real model-backed AI-generated text detection service running on CPU.
    Uses lazy loading and singleton caching.
    Strictly adheres to forensic guardrails: never presents raw logits as calibrated
    ground truth and ensures outputs are never the sole basis for forensic findings.
    """

    _model = None
    _tokenizer = None
    _load_attempted = False
    _model_name = "roberta-base-openai-detector"

    @classmethod
    def _load_model(cls) -> bool:
        if cls._load_attempted:
            return cls._model is not None

        cls._load_attempted = True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            # Load strictly from local cache to prevent unexpected network hangs or failures
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name, local_files_only=True)
            cls._model = AutoModelForSequenceClassification.from_pretrained(cls._model_name, local_files_only=True)
            cls._model.eval()
            return True
        except Exception as e:
            logger.warning(f"Could not load local AI text detector '{cls._model_name}': {e}")
            cls._model = None
            cls._tokenizer = None
            return False

    @classmethod
    def detect_ai_text(cls, text: str) -> Dict[str, Any]:
        """
        Runs inference on CPU if local model is available, otherwise returns safe skipped/not_implemented status.
        """
        if not text or len(text.strip()) < 20:
            return {
                "status": "skipped",
                "model_name": None,
                "model_version_or_source": None,
                "input_requirements": {},
                "raw_output": None,
                "raw_scores": {},
                "inference_ran": False,
                "errors": [],
                "limitations": ["Text is too short for reliable statistical AI detection (< 20 characters)."]
            }

        is_loaded = cls._load_model()

        if not is_loaded:
            return {
                "status": "not_implemented",
                "model_name": cls._model_name,
                "model_version_or_source": None,
                "input_requirements": {"max_sequence_length": 512, "device": "cpu"},
                "raw_output": None,
                "raw_scores": {},
                "inference_ran": False,
                "errors": [],
                "limitations": [
                    f"Pretrained detector '{cls._model_name}' is not locally available in current environment.",
                    "No synthetic scores were fabricated in adherence to forensic integrity guidelines."
                ]
            }

        try:
            import torch
            inputs = cls._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )

            with torch.no_grad():
                outputs = cls._model(**inputs)
                logits = outputs.logits[0].tolist()
                probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

            # Roberta-base-openai-detector labels: index 0 = Real, index 1 = Fake (AI-generated)
            # Or config id2label if available
            id2label = getattr(cls._model.config, "id2label", {0: "Real", 1: "Fake"})
            label_0 = id2label.get(0, "Real")
            label_1 = id2label.get(1, "Fake")

            raw_scores = {
                label_0: round(float(probs[0]), 4),
                label_1: round(float(probs[1]), 4)
            }

            return {
                "status": "completed",
                "model_name": cls._model_name,
                "model_version_or_source": "HuggingFace local cache (roberta-base-openai-detector)",
                "input_requirements": {
                    "max_sequence_length": 512,
                    "truncated": len(cls._tokenizer.encode(text)) > 512,
                    "device": "cpu"
                },
                "raw_output": {
                    "logits": [round(l, 4) for l in logits]
                },
                "raw_scores": raw_scores,
                "inference_ran": True,
                "errors": [],
                "limitations": [
                    "Detector output reflects statistical token sequence likelihood under the RoBERTa OpenAI detector model.",
                    "Raw softmax scores represent uncalibrated heuristic indicators and must not be interpreted as absolute proof of AI generation.",
                    "Model output is an auxiliary investigative indicator and must never be the sole basis for a forensic finding."
                ]
            }
        except Exception as e:
            logger.error(f"Error during AI text detection inference: {e}")
            return {
                "status": "failed",
                "model_name": cls._model_name,
                "model_version_or_source": "HuggingFace local cache",
                "input_requirements": {"max_sequence_length": 512, "device": "cpu"},
                "raw_output": None,
                "raw_scores": {},
                "inference_ran": False,
                "errors": [f"Inference exception: {str(e)}"],
                "limitations": ["Inference failed during evaluation; no fabricated scores generated."]
            }

