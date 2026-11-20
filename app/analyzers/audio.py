"""Evidence-preserving, signal-level audio analysis.

The analyzer consumes the existing eight-field Analyzer Input Contract. It
does not assign evidence/signal/finding IDs and never writes to source audio.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from app.analyzers.config import AudioAnalyzerSettings, SETTINGS
from app.analyzers.audio_models import transcribe, detect_synthetic_audio

CONTRACT_FIELDS = {
    "evidence_id", "file_path", "original_filename", "file_type",
    "mime_type", "file_size", "sha256", "ingestion_metadata",
}
SUPPORTED_AUDIO = {".wav": {"audio/wav", "audio/x-wav"}, ".mp3": {"audio/mpeg", "audio/mp3"}, ".m4a": {"audio/mp4", "audio/x-m4a"}}
ANALYZER_NAME = "TruthLens Audio Signal Analyzer"
ANALYZER_VERSION = "1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _dbfs(value: float) -> float | None:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _merge_ranges(mask: np.ndarray, hop_s: float, frame_s: float, minimum: float) -> list[dict[str, float]]:
    ranges: list[dict[str, float]] = []
    start = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            begin = start * hop_s
            end = (index - 1) * hop_s + frame_s
            if end - begin >= minimum:
                ranges.append({"start_seconds": begin, "end_seconds": end, "duration_seconds": end - begin})
            start = None
    return ranges


def _voiced(frame: np.ndarray, sample_rate: int, settings: AudioAnalyzerSettings) -> tuple[float | None, float]:
    frame = frame.astype(np.float64, copy=False)
    frame = frame - frame.mean()
    if np.max(np.abs(frame), initial=0.0) < 1e-5:
        return None, 0.0
    size = 1 << (2 * len(frame) - 1).bit_length()
    spectrum = np.fft.rfft(frame, n=size)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:len(frame)]
    if autocorrelation[0] <= 0:
        return None, 0.0
    min_lag = max(1, int(sample_rate / settings.voiced_f0_max_hz))
    max_lag = min(len(autocorrelation) - 1, int(sample_rate / settings.voiced_f0_min_hz))
    if max_lag <= min_lag:
        return None, 0.0
    section = autocorrelation[min_lag:max_lag + 1]
    best_lag = min_lag + int(np.argmax(section))
    periodicity = float(autocorrelation[best_lag] / autocorrelation[0])
    pitch = float(sample_rate / best_lag) if periodicity >= settings.voiced_periodicity_min else None
    return pitch, max(0.0, min(1.0, periodicity))


def _read_wav(path: Path, settings: AudioAnalyzerSettings) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frame_count = source.getnframes()
        if source.getcomptype() != "NONE":
            raise ValueError(f"Unsupported WAV compression: {source.getcomptype()}")
        if frame_count <= 0 or sample_rate <= 0 or channels <= 0:
            raise ValueError("WAV has no readable audio stream.")
        if frame_count * channels > settings.max_decoded_samples:
            raise ValueError("Decoded audio exceeds the configured analyzer sample bound.")
        raw = source.readframes(frame_count)
    if sample_width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        bit_depth = 8
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        bit_depth = 16
    elif sample_width == 3:
        bytes3 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        integers = bytes3[:, 0] | (bytes3[:, 1] << 8) | (bytes3[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        values = integers.astype(np.float32) / 8388608.0
        bit_depth = 24
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        bit_depth = 32
    else:
        raise ValueError(f"Unsupported PCM bit depth: {sample_width * 8}.")
    return values.reshape(-1, channels), sample_rate, channels, bit_depth


def _read_compressed(path: Path, settings: AudioAnalyzerSettings) -> tuple[np.ndarray, int, int, int | None, str]:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("FFmpeg decoder is unavailable; MP3/M4A cannot be analyzed.") from exc
    sample_rate = settings.analysis_sample_rate
    # Bound FFmpeg's decoded stream before collecting stdout. The cap is applied
    # in the output sample domain, including a small sentinel margin.
    max_values = settings.max_decoded_samples
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(sample_rate), "-ac", "1", "-t", str(max_values / sample_rate + 0.1), "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        raw, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate()
        raise ValueError("Audio decode exceeded the 45-second analysis time limit.")
    if len(raw) > max_values * 4:
        raise ValueError("Decoded audio exceeds the configured analyzer sample bound.")
    if process.returncode != 0 or not raw:
        error = stderr.decode("utf-8", errors="replace")[-1500:]
        raise ValueError(f"Audio decode failed: {error or 'no decoded samples'}")
    if len(raw) % 4:
        raise ValueError("Audio decoder returned an incomplete PCM sample.")
    samples = np.frombuffer(raw, dtype="<f4").copy()
    channels = 1
    # Compressed inputs are deterministically downmixed for analysis; source
    # channel count is reported from metadata where available.
    source_channels = 1
    try:
        from mutagen import File as MutagenFile
        container = MutagenFile(str(path))
        source_channels = int(getattr(getattr(container, "info", None), "channels", 1) or 1)
    except Exception:
        pass
    return samples.reshape(-1, 1), sample_rate, source_channels, None, "ffmpeg"


class AudioAnalyzer:
    """Reusable audio analyzer; independent of FastAPI and Streamlit."""

    def __init__(self, artifact_root: str | Path = "data/artifacts", settings: AudioAnalyzerSettings = SETTINGS):
        self.artifact_root = Path(artifact_root).resolve()
        self.settings = settings

    def analyze(self, analyzer_input: dict[str, Any], *, run_transcription: bool = False,
                run_synthetic_detection: bool = False, language: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        started_at = _utc_now()
        evidence_id = str(analyzer_input.get("evidence_id", "unknown"))
        result = {
            "evidence_id": evidence_id,
            "analysis_id": None,
            "analysis_timestamp": started_at,
            "analyzer": {"name": ANALYZER_NAME, "version": ANALYZER_VERSION},
            "analysis_status": "FAILED",
            "execution_metadata": {"started_at": started_at, "duration_ms": None},
            "audio_metadata": {}, "signals": [], "findings": [], "timeline_segments": [],
            "artifacts": [], "model_info": {"name": "Wav2Vec2 audio spoof classifier (ONNX int4)",
                "version": SETTINGS.deepfake_model_revision, "available": False,
                "status": "skipped", "explanation": "Detector was not requested; model weights load only on explicit request."},
            "speech_detection": {"status": "pending", "method": "heuristic periodicity and frame-energy analysis"},
            "transcription": {"status": "skipped", "model": SETTINGS.transcription_model, "text": "", "segments": [], "language": None,
                              "error": None, "reason": "Transcription was not requested."},
            "synthetic_audio_detection": {"status": "skipped", "model_id": SETTINGS.deepfake_model, "prediction": None,
                                          "reason": "Synthetic-audio detection was not requested."},
            "voice_characteristics": {"status": "pending"},
            "modules": {},
            "limitations": [
                "Signal indicators are measurable observations and do not establish authenticity or manipulation.",
                "Voiced activity is heuristic; it is not transcription, speaker identification, or reliable speech recognition.",
                "Speaking rate, language, speaker identity, and definitive edit/deepfake attribution are not assessed.",
                "Noise-floor and cutoff indicators can reflect ordinary microphones, codecs, rooms, or resampling.",
            ],
            "errors": [], "warnings": [],
        }
        try:
            self._validate_contract(analyzer_input)
            path = Path(analyzer_input["file_path"])
            decoded = self._decode(path)
            samples, sample_rate, channels, bit_depth, codec, decode_method = decoded
            if samples.size == 0 or sample_rate <= 0:
                raise ValueError("Decoded audio contains no samples.")
            if samples.ndim != 2 or samples.shape[1] != channels:
                raise ValueError("Decoded audio channel layout is inconsistent.")
            if not np.all(np.isfinite(samples)):
                raise ValueError("Decoded audio contains NaN or infinite samples.")
            samples = np.clip(samples, -1.0, 1.0)
            result["audio_metadata"] = self._analyze_signal(samples, sample_rate, channels, bit_depth, codec, analyzer_input, decode_method, result)
            metrics = result["audio_metadata"]
            result["speech_detection"] = {
                "status": "completed", "method": "heuristic periodicity/F0 with frame-energy gating",
                "voiced_activity_candidates": metrics["voiced_activity_candidate_regions"],
                "interpretation": "Voiced-activity candidates only; this is not speech recognition.",
            }
            pitch = metrics["estimated_f0_hz"]
            pitch_values = pitch.get("median")
            result["voice_characteristics"] = {
                "status": "completed" if pitch_values is not None else "unavailable",
                "fundamental_frequency_hz": pitch,
                "pitch_standard_deviation_hz": metrics.get("pitch_standard_deviation_hz"),
                "voiced_activity_regions": metrics["voiced_activity_candidate_regions"],
                "interpretation": "Acoustic pitch characteristics from periodic frames; not speaker identification or proof of a human voice.",
                "speaking_rate": None,
                "speaker_identity": "not assessed",
            }
            result["modules"] = {
                "signal_forensics": {"status": "completed"},
                "speech_detection": {"status": "completed", "kind": "heuristic"},
                "transcription": {"status": "skipped" if not run_transcription else "pending"},
                "synthetic_audio_detection": {"status": "skipped" if not run_synthetic_detection else "pending"},
                "voice_characteristics": {"status": result["voice_characteristics"]["status"], "kind": "heuristic"},
                "artifact_generation": {"status": "pending"},
            }
            mono = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
            if run_transcription:
                try:
                    transcription = transcribe(mono, sample_rate, language=language)
                except Exception as exc:
                    transcription = {"status": "failed", "text": "", "segments": [], "language": None,
                                     "error": f"Transcription module failed: {exc}"}
                result["transcription"] = transcription
                result["modules"]["transcription"] = {"status": transcription["status"]}
                if transcription.get("error"):
                    result["errors"].append({"stage": "TRANSCRIPTION", "message": transcription["error"]})
                for segment in transcription.get("segments", []):
                    result["timeline_segments"].append({
                        "start_seconds": segment["start_seconds"], "end_seconds": segment["end_seconds"],
                        "duration_seconds": segment["end_seconds"] - segment["start_seconds"],
                        "segment_type": "speech", "indicator": "TRANSCRIPTION_AVAILABLE",
                        "description": segment["text"], "supporting_measurements": {},
                        "explanation": "Timestamped segment returned by the speech recognition model.",
                        "reliability": "model output",
                    })
                if transcription.get("text"):
                    index = len(result["signals"])
                    result["signals"].append({"signal_type": "TRANSCRIPTION_AVAILABLE",
                                              "description": "Speech transcription was returned by the configured local model.",
                                              "measurement": {"segment_count": len(transcription.get("segments", [])),
                                                              "language": transcription.get("language")},
                                              "reliability": "model output"})
                    result["findings"].append({"description": "Timestamped transcription is available for review.",
                                               "supporting_signal_indices": [index], "severity": "informational",
                                               "uncertainty": "Automatic speech recognition can omit, alter, or hallucinate words, especially in noise or unsupported domains."})
            if run_synthetic_detection:
                try:
                    detection = detect_synthetic_audio(samples, sample_rate)
                except Exception as exc:
                    detection = {"status": "failed", "prediction": None,
                                 "error": f"Synthetic-audio detector module failed: {exc}"}
                result["synthetic_audio_detection"] = detection
                result["modules"]["synthetic_audio_detection"] = {"status": detection["status"]}
                result["model_info"] = {"name": detection.get("detector_name"),
                                         "version": detection.get("model_version"),
                                         "available": detection["status"] == "completed",
                                         "status": detection["status"], "explanation": detection.get("error")}
                if detection["status"] == "completed":
                    index = len(result["signals"])
                    result["signals"].append({"signal_type": "SYNTHETIC_AUDIO_MODEL_INDICATOR",
                                              "description": "A pretrained audio classifier returned an uncalibrated spoof-classification output.",
                                              "measurement": {"prediction": detection.get("prediction"),
                                                              "class_scores": detection.get("class_scores"),
                                                              "raw_logits": detection.get("raw_logits")},
                                              "reliability": "uncalibrated model output"})
                    result["findings"].append({"description": "Synthetic-audio model indicator available for investigator review.",
                                               "supporting_signal_indices": [index], "severity": "informational",
                                               "uncertainty": "The model output is not calibrated and is not proof of authenticity or manipulation."})
                    result["timeline_segments"].append({
                        "start_seconds": 0.0, "end_seconds": float(metrics["duration_seconds"]),
                        "duration_seconds": float(metrics["duration_seconds"]),
                        "segment_type": "synthetic_audio_model_indicator",
                        "indicator": "SYNTHETIC_AUDIO_MODEL_INDICATOR",
                        "description": f"Whole-clip model class output: {detection.get('prediction')}.",
                        "supporting_measurements": {"class_scores": detection.get("class_scores")},
                        "explanation": "Uncalibrated classification output for the whole clip; not a verdict.",
                        "reliability": "uncalibrated model output", "analysis_source": detection.get("model_id"),
                    })
                elif detection.get("error"):
                    result["errors"].append({"stage": "SYNTHETIC_AUDIO_DETECTION", "message": detection["error"]})
            for item in result["timeline_segments"]:
                item.setdefault("evidence_id", evidence_id)
                item.setdefault("analysis_source", item.get("source") or item.get("segment_type", "signal_forensics"))
            result["timeline_segments"].sort(key=lambda item: item.get("start_seconds", 0))
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
            result["analysis_id"] = run_id
            out_dir = self.artifact_root / evidence_id / "audio" / run_id
            out_dir.mkdir(parents=True, exist_ok=False)
            try:
                self._write_artifacts(out_dir, run_id, evidence_id, samples, sample_rate, result)
                result["modules"]["artifact_generation"] = {"status": "completed"}
            except Exception as exc:
                result["modules"]["artifact_generation"] = {"status": "failed", "explanation": str(exc)}
                result["errors"].append({"stage": "ARTIFACT_GENERATION", "message": str(exc)})
            if result["transcription"].get("status") == "completed":
                transcript_name = f"{evidence_id}_audio_{run_id}_transcript.json"
                try:
                    (out_dir / transcript_name).write_text(json.dumps(result["transcription"], indent=2, ensure_ascii=False), encoding="utf-8")
                    result["artifacts"].append(self._artifact(out_dir, transcript_name, "application/json", "Model transcription and timestamped segments."))
                except OSError as exc:
                    result["modules"]["artifact_generation"] = {"status": "failed", "explanation": str(exc)}
                    result["errors"].append({"stage": "TRANSCRIPT_ARTIFACT", "message": str(exc)})
            optional_statuses = [result["modules"].get(key, {}).get("status")
                                 for key in ("transcription", "synthetic_audio_detection")]
            result["analysis_status"] = "PARTIAL" if any(
                status in {"failed", "unavailable", "unsupported"} for status in optional_statuses
            ) or result["modules"].get("artifact_generation", {}).get("status") == "failed" else "COMPLETED"
            result["execution_metadata"].update(run_id=run_id, artifact_directory=str(out_dir))
            (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            result["artifacts"].append(self._artifact(out_dir, "analysis.json", "application/json", "Structured audio analyzer output."))
            latest_path = self.artifact_root / evidence_id / "audio" / "latest.json"
            latest_path.write_text(json.dumps({"run_id": run_id, "result_path": str(out_dir / "analysis.json")}, indent=2), encoding="utf-8")
        except Exception as exc:
            result["analysis_status"] = "FAILED"
            result["errors"].append({"stage": "VALIDATION_OR_ANALYSIS", "message": str(exc)})
        finally:
            result["execution_metadata"]["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def _validate_contract(self, data: dict[str, Any]) -> None:
        if set(data) != CONTRACT_FIELDS:
            raise ValueError("Audio Analyzer requires the exact eight-field Analyzer Input Contract.")
        if data["file_type"] != "audio":
            raise ValueError(f"Audio Analyzer accepts file_type 'audio'; received {data['file_type']!r}.")
        path = Path(str(data["file_path"]))
        if not path.is_file():
            raise ValueError("Original audio evidence file is missing.")
        extension = path.suffix.lower()
        if extension not in SUPPORTED_AUDIO:
            raise ValueError(f"Unsupported audio extension: {extension or '(none)'}")
        if data["mime_type"] not in SUPPORTED_AUDIO[extension]:
            raise ValueError(f"MIME type {data['mime_type']!r} does not match {extension} audio.")
        size = path.stat().st_size
        if size <= 0 or size != int(data["file_size"]):
            raise ValueError("Audio file size does not match the Analyzer Input Contract.")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != str(data["sha256"]).lower():
            raise ValueError("Audio file integrity check failed against SHA-256.")

    def _decode(self, path: Path) -> tuple[np.ndarray, int, int, int | None, str, str]:
        if path.suffix.lower() == ".wav":
            samples, sample_rate, channels, bit_depth = _read_wav(path, self.settings)
            return samples, sample_rate, channels, bit_depth, "PCM", "python-wave"
        samples, sample_rate, channels, bit_depth, decoder = _read_compressed(path, self.settings)
        return samples, sample_rate, channels, bit_depth, path.suffix[1:].upper(), decoder

    def _analyze_signal(self, samples: np.ndarray, sample_rate: int, channels: int,
                        bit_depth: int | None, codec: str, contract: dict, decoder: str,
                        result: dict) -> dict[str, Any]:
        settings = self.settings
        mono = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
        duration = len(mono) / sample_rate
        frame_size = max(16, int(sample_rate * settings.frame_seconds))
        hop_size = max(1, int(sample_rate * settings.hop_seconds))
        if len(mono) < frame_size:
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[:len(mono)] = mono
            mono_for_frames = padded
        else:
            mono_for_frames = mono
        frame_starts = np.arange(0, max(1, len(mono_for_frames) - frame_size + 1), hop_size, dtype=np.int64)
        if frame_starts.size == 0:
            frame_starts = np.array([0], dtype=np.int64)
        window = np.hanning(frame_size).astype(np.float32)
        n_fft = 1 << (frame_size - 1).bit_length()
        frame_rms = np.empty(len(frame_starts), dtype=np.float64)
        frame_peak = np.empty(len(frame_starts), dtype=np.float64)
        centroid = np.empty(len(frame_starts), dtype=np.float64)
        flatness = np.empty(len(frame_starts), dtype=np.float64)
        rolloff = np.empty(len(frame_starts), dtype=np.float64)
        f0 = np.full(len(frame_starts), np.nan, dtype=np.float64)
        periodicity = np.zeros(len(frame_starts), dtype=np.float64)
        spectra: list[np.ndarray] = []
        frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        for index, start in enumerate(frame_starts):
            frame = mono_for_frames[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            frame_rms[index] = np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
            frame_peak[index] = np.max(np.abs(frame), initial=0.0)
            magnitude = np.abs(np.fft.rfft(frame * window, n=n_fft)).astype(np.float64) + 1e-12
            spectra.append(magnitude)
            total = magnitude.sum()
            centroid[index] = float(np.sum(frequencies * magnitude) / total)
            flatness[index] = float(np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude))
            cumulative = np.cumsum(magnitude)
            rolloff[index] = float(frequencies[min(np.searchsorted(cumulative, cumulative[-1] * 0.85), len(frequencies) - 1)])
            if frame_rms[index] > 10 ** (settings.silence_threshold_dbfs / 20.0):
                f0[index], periodicity[index] = _voiced(frame, sample_rate, settings)
        spectrogram = np.stack(spectra, axis=1)
        peak = float(np.max(np.abs(samples), initial=0.0))
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        clip_ratio = float(np.mean(np.abs(samples) >= settings.clipping_threshold))
        silence_mask = frame_rms <= 10 ** (settings.silence_threshold_dbfs / 20.0)
        silence_ranges = _merge_ranges(silence_mask, hop_size / sample_rate, frame_size / sample_rate, settings.silence_min_seconds)
        active_mask = ~silence_mask
        activity_ranges = _merge_ranges(active_mask, hop_size / sample_rate, frame_size / sample_rate, 0.10)
        voiced_mask = np.isfinite(f0) & (periodicity >= settings.voiced_periodicity_min)
        voiced_ranges = _merge_ranges(voiced_mask, hop_size / sample_rate, frame_size / sample_rate, 0.15)

        frequency_power = np.mean(np.square(spectrogram), axis=1)
        def band_energy(low: float, high: float) -> float:
            selected = (frequencies >= low) & (frequencies < min(high, sample_rate / 2))
            return float(np.sum(frequency_power[selected]))
        nyquist = sample_rate / 2
        bands = {
            "0_250_hz": band_energy(0, 250), "250_2000_hz": band_energy(250, 2000),
            "2000_8000_hz": band_energy(2000, 8000), "8000_hz_to_nyquist": band_energy(8000, nyquist + 1),
        }
        total_band_power = sum(bands.values())
        band_ratios = {key: (value / total_band_power if total_band_power > 0 else None) for key, value in bands.items()}
        active_rms = frame_rms[active_mask]
        noise_floor = float(np.percentile(active_rms, 10)) if active_rms.size else 0.0
        rms_db = 20 * np.log10(np.maximum(frame_rms, 1e-12))
        rms_percentiles = {f"p{p:02d}_dbfs": float(np.percentile(rms_db, p)) for p in (10, 50, 90, 95, 99)}
        dynamic_range = rms_percentiles["p95_dbfs"] - rms_percentiles["p10_dbfs"]
        discontinuities = self._find_discontinuities(frame_starts, hop_size, sample_rate, rms_db, centroid, flatness)
        silence_ratio = float(np.mean(silence_mask))
        mean_spectrum = np.mean(spectrogram, axis=1)
        mid_energy = band_energy(0.30 * nyquist, 0.70 * nyquist)
        upper_energy = band_energy(0.85 * nyquist, nyquist + 1)
        upper_ratio = upper_energy / max(mid_energy, 1e-18)
        metrics = {
            "duration_seconds": duration, "sample_rate": sample_rate,
            "source_sample_rate": contract.get("ingestion_metadata", {}).get("sample_rate"),
            "analysis_sample_rate": sample_rate, "channel_count": channels,
            "bit_depth": bit_depth, "codec": codec, "decode_method": decoder,
            "file_size": contract["file_size"], "peak_amplitude": peak,
            "peak_dbfs": _dbfs(peak), "rms_amplitude": rms, "rms_dbfs": _dbfs(rms),
            "clipping_sample_ratio": clip_ratio,
            "clipping_detected": clip_ratio >= settings.clipping_min_ratio,
            "amplitude_distribution": {f"p{p:02d}_absolute_amplitude": float(np.percentile(np.abs(samples), p)) for p in (1, 50, 95, 99)},
            "frame_rms_percentiles_dbfs": rms_percentiles,
            "dynamic_range_db": float(dynamic_range), "silence_ratio": silence_ratio,
            "silence_regions": silence_ranges, "sound_activity_regions": activity_ranges,
            "voiced_activity_candidate_regions": voiced_ranges,
            "estimated_f0_hz": {"minimum": float(np.nanmin(f0)) if np.isfinite(f0).any() else None,
                                 "median": float(np.nanmedian(f0)) if np.isfinite(f0).any() else None,
                                 "maximum": float(np.nanmax(f0)) if np.isfinite(f0).any() else None},
            "pitch_standard_deviation_hz": float(np.nanstd(f0)) if np.isfinite(f0).any() else None,
            "spectral_centroid_hz_mean": float(np.mean(centroid)),
            "spectral_rolloff_85_hz_mean": float(np.mean(rolloff)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "frequency_band_energy": bands, "frequency_band_energy_ratio": band_ratios,
            "high_frequency_energy_ratio": float(upper_ratio),
            "high_frequency_cutoff_indicator": bool(sample_rate >= 32_000 and upper_ratio < settings.high_frequency_floor_ratio),
            "estimated_noise_floor_dbfs": _dbfs(noise_floor),
            "noise_floor_variation_db": float(np.percentile(20 * np.log10(np.maximum(active_rms, 1e-12)), 90) - np.percentile(20 * np.log10(np.maximum(active_rms, 1e-12)), 10)) if active_rms.size else None,
            "snr_db": None, "snr_unavailable_reason": "No calibrated noise-only reference segment was available; an SNR estimate would be unsupported.",
            "amplitude_discontinuity_count": int(np.sum(np.abs(np.diff(mono)) >= 0.9)),
            "spectral_discontinuity_count": len(discontinuities),
        }
        result["timeline_segments"] = self._make_timeline(silence_ranges, activity_ranges, voiced_ranges, discontinuities)
        self._make_signals_and_findings(result, metrics, discontinuities, upper_ratio)
        result["_render_data"] = {"mono": mono, "spectrogram": spectrogram, "frequencies": frequencies,
                                  "mean_spectrum": mean_spectrum, "frame_hop_seconds": hop_size / sample_rate}
        return metrics

    def _find_discontinuities(self, frame_starts: np.ndarray, hop: int, sample_rate: int,
                             rms_db: np.ndarray, centroid: np.ndarray, flatness: np.ndarray) -> list[dict[str, Any]]:
        if len(rms_db) < 4:
            return []
        diff_rms = np.abs(np.diff(rms_db))
        diff_centroid = np.abs(np.diff(centroid)) / max(float(np.median(centroid)), 1.0)
        diff_flatness = np.abs(np.diff(flatness))
        def robust_threshold(values: np.ndarray) -> float:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            return median + self.settings.discontinuity_mad_multiplier * max(mad, 1e-6)
        hit = (diff_rms > robust_threshold(diff_rms)) & ((diff_centroid > robust_threshold(diff_centroid)) | (diff_flatness > robust_threshold(diff_flatness)))
        indices = np.flatnonzero(hit)
        return [{"time_seconds": float(frame_starts[i + 1] / sample_rate),
                 "rms_change_db": float(diff_rms[i]), "relative_centroid_change": float(diff_centroid[i]),
                 "flatness_change": float(diff_flatness[i])} for i in indices]

    def _make_timeline(self, silence: list[dict], activity: list[dict], voiced: list[dict], discontinuities: list[dict]) -> list[dict]:
        segments: list[dict[str, Any]] = []
        for kind, ranges, explanation in (
            ("silence", silence, "Frame RMS was at or below the configured silence threshold."),
            ("sound_activity", activity, "Audio energy exceeded the configured silence threshold; this is not a speech transcription."),
            ("voiced_activity_candidate", voiced, "Heuristic periodicity/F0 suggests voiced activity; this is not speaker identification."),
        ):
            for region in ranges:
                segments.append({**region, "segment_type": kind, "indicator": kind,
                                 "supporting_measurements": {"method": "frame RMS" if kind != "voiced_activity_candidate" else "autocorrelation pitch/periodicity"},
                                 "explanation": explanation, "reliability": "heuristic" if kind == "voiced_activity_candidate" else "measurement-based"})
        for item in discontinuities:
            t = item["time_seconds"]
            segments.append({"start_seconds": t, "end_seconds": t, "duration_seconds": 0.0,
                             "segment_type": "potential_acoustic_transition", "indicator": "POSSIBLE_EDIT_BOUNDARY",
                             "supporting_measurements": item,
                             "explanation": "Simultaneous robust changes in level and spectral features; this is a transition indicator, not proof of editing.",
                             "reliability": "heuristic"})
        # Active frames without a voiced-periodicity candidate remain unknown;
        # they may be noise, music, unvoiced speech, or other sound.
        for region in self._unclassified_activity(activity, voiced):
            segments.append({**region, "segment_type": "unknown_audio_activity",
                             "indicator": "AUDIO_ACTIVITY_UNCLASSIFIED",
                             "supporting_measurements": {"method": "frame RMS; no voiced candidate overlap"},
                             "explanation": "Audible activity without a voiced-activity candidate; content is not classified as speech or non-speech.",
                             "reliability": "measurement-based activity; content unknown"})
        return sorted(segments, key=lambda entry: entry["start_seconds"])

    @staticmethod
    def _unclassified_activity(activity: list[dict], voiced: list[dict]) -> list[dict]:
        remaining = []
        for active in activity:
            pieces = [(float(active["start_seconds"]), float(active["end_seconds"]))]
            for candidate in voiced:
                next_pieces = []
                for start, end in pieces:
                    cut_start, cut_end = float(candidate["start_seconds"]), float(candidate["end_seconds"])
                    if cut_end <= start or cut_start >= end:
                        next_pieces.append((start, end))
                    else:
                        if cut_start > start:
                            next_pieces.append((start, min(cut_start, end)))
                        if cut_end < end:
                            next_pieces.append((max(cut_end, start), end))
                pieces = next_pieces
            for start, end in pieces:
                if end > start:
                    remaining.append({"start_seconds": start, "end_seconds": end,
                                     "duration_seconds": end - start})
        return remaining

    def _make_signals_and_findings(self, result: dict, metrics: dict,
                                   discontinuities: list[dict], upper_ratio: float) -> None:
        signals: list[dict[str, Any]] = []
        def emit(signal_type: str, description: str, measurement: dict, reliability: str = "measurement-based", time_seconds=None):
            signal = {"signal_type": signal_type, "description": description,
                      "measurement": measurement, "reliability": reliability}
            if time_seconds is not None:
                signal["time_seconds"] = time_seconds
            signals.append(signal)
        emit("AUDIO_LEVELS_MEASURED", "Peak, RMS, distribution, and clipping measurements computed from decoded samples.",
             {key: metrics[key] for key in ("peak_amplitude", "rms_amplitude", "peak_dbfs", "rms_dbfs", "clipping_sample_ratio")})
        if metrics["clipping_detected"]:
            emit("AUDIO_CLIPPING_DETECTED", "The measured sample fraction at or beyond the configured clipping threshold exceeded the configured minimum.",
                 {"clipping_sample_ratio": metrics["clipping_sample_ratio"], "threshold": self.settings.clipping_threshold})
        if metrics["silence_ratio"] >= self.settings.high_silence_ratio:
            emit("HIGH_SILENCE_RATIO", "A high proportion of analyzed frames fell below the configured silence threshold.",
                 {"silence_ratio": metrics["silence_ratio"], "threshold_dbfs": self.settings.silence_threshold_dbfs})
        if metrics["voiced_activity_candidate_regions"]:
            emit("SPEECH_ACTIVITY_CANDIDATE", "Periodic voiced activity candidates were detected by a simple F0 heuristic; speech is not transcribed.",
                 {"candidate_region_count": len(metrics["voiced_activity_candidate_regions"]), "f0_hz": metrics["estimated_f0_hz"]}, "heuristic")
        if discontinuities:
            emit("SPECTRAL_DISCONTINUITY", "Abrupt joint level/spectral feature changes were detected using a robust median/MAD threshold.",
                 {"count": len(discontinuities), "events": discontinuities}, "heuristic")
            emit("POSSIBLE_EDIT_BOUNDARY", "Acoustic transitions may merit review; ordinary scene, microphone, or codec changes can produce the same pattern.",
                 {"candidate_count": len(discontinuities)}, "heuristic")
        if metrics["high_frequency_cutoff_indicator"]:
            emit("UNUSUAL_FREQUENCY_CUTOFF", "Upper-band energy was very low relative to mid-band energy.",
                 {"upper_to_mid_energy_ratio": upper_ratio, "configured_floor_ratio": self.settings.high_frequency_floor_ratio}, "heuristic")
        result["signals"] = signals
        findings = []
        for index, signal in enumerate(signals):
            if signal["signal_type"] == "AUDIO_CLIPPING_DETECTED":
                findings.append({"description": "Clipping-level samples were measured in the decoded audio.", "supporting_signal_indices": [index], "severity": "low", "uncertainty": "This describes waveform saturation only; it does not establish manipulation."})
            elif signal["signal_type"] == "HIGH_SILENCE_RATIO":
                findings.append({"description": "A high silence-threshold ratio was measured.", "supporting_signal_indices": [index], "severity": "informational", "uncertainty": "Silence can be expected in ordinary recordings."})
            elif signal["signal_type"] == "POSSIBLE_EDIT_BOUNDARY":
                findings.append({"description": "One or more abrupt acoustic feature transitions may warrant listening review.", "supporting_signal_indices": [index], "severity": "informational", "uncertainty": "Transitions can result from normal content, recording conditions, or encoding."})
        result["findings"] = findings

    def _write_artifacts(self, out_dir: Path, run_id: str, evidence_id: str,
                         samples: np.ndarray, sample_rate: int, result: dict) -> None:
        render = result.pop("_render_data")
        run_tag = f"{evidence_id}_audio_{run_id}"
        waveform = self._waveform_image(render["mono"])
        spectrogram = self._spectrogram_image(render["spectrogram"])
        spectrum = self._spectrum_image(render["frequencies"], render["mean_spectrum"])
        timeline = self._timeline_image(result["timeline_segments"], len(render["mono"]) / sample_rate)
        for filename, image, description in (
            (f"{run_tag}_waveform.png", waveform, "Measured waveform overview."),
            (f"{run_tag}_spectrogram.png", spectrogram, "Magnitude spectrogram derived from short-time FFT frames."),
            (f"{run_tag}_frequency_spectrum.png", spectrum, "Mean magnitude frequency spectrum."),
            (f"{run_tag}_timeline.png", timeline, "Silence, voiced-activity candidates, and acoustic transitions."),
        ):
            image.save(out_dir / filename, format="PNG", optimize=True)
            result["artifacts"].append(self._artifact(out_dir, filename, "image/png", description))
        result["audio_metadata"]["artifact_summary"] = {
            "waveform": "peak envelope by horizontal sample bins",
            "spectrogram": "log magnitude from short-time FFT; displayed scale is normalized for visualization",
            "frequency_spectrum": "mean FFT magnitude per frequency bin",
            "timeline": "detected analysis segments mapped to the original audio duration",
        }

    @staticmethod
    def _artifact(directory: Path, filename: str, media_type: str, description: str) -> dict[str, str]:
        return {"name": filename, "media_type": media_type, "path": str(directory / filename), "description": description}

    @staticmethod
    def _waveform_image(samples: np.ndarray, width: int = 1200, height: int = 300) -> Image.Image:
        image = Image.new("RGB", (width, height), "#101827")
        draw = ImageDraw.Draw(image)
        center = height // 2
        draw.line((40, center, width - 20, center), fill="#64748b", width=1)
        bins = np.array_split(samples, width - 60)
        for x, chunk in enumerate(bins, start=40):
            if chunk.size:
                lo = float(np.min(chunk)); hi = float(np.max(chunk))
                draw.line((x, center - hi * (height * .42), x, center - lo * (height * .42)), fill="#4ade80", width=1)
        draw.text((12, 10), "Waveform · normalized amplitude", fill="#e2e8f0")
        return image

    @staticmethod
    def _spectrogram_image(spectrogram: np.ndarray, width: int = 1200, height: int = 420) -> Image.Image:
        decimated = spectrogram[::max(1, spectrogram.shape[0] // height), ::max(1, spectrogram.shape[1] // width)]
        log_values = 20 * np.log10(np.maximum(decimated, 1e-10))
        low, high = np.percentile(log_values, (5, 99.5))
        scaled = np.clip((log_values - low) / max(high - low, 1e-9) * 255, 0, 255).astype(np.uint8)
        gray = Image.fromarray(np.flipud(scaled), mode="L").resize((width, height))
        return Image.merge("RGB", (gray.point(lambda v: min(255, int(v * .75))), gray, gray.point(lambda v: min(255, int(v * .45)))))

    @staticmethod
    def _spectrum_image(frequencies: np.ndarray, magnitude: np.ndarray, width: int = 1200, height: int = 300) -> Image.Image:
        image = Image.new("RGB", (width, height), "#101827")
        draw = ImageDraw.Draw(image)
        values = 20 * np.log10(np.maximum(magnitude, 1e-10))
        sampled = values[::max(1, len(values) // (width - 80))]
        lo, hi = float(np.percentile(sampled, 2)), float(np.percentile(sampled, 99))
        points = []
        for i, value in enumerate(sampled):
            x = 50 + i * (width - 70) / max(1, len(sampled) - 1)
            y = height - 30 - np.clip((value - lo) / max(hi - lo, 1e-9), 0, 1) * (height - 70)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill="#60a5fa", width=2)
        draw.text((12, 10), "Mean frequency spectrum · dB relative", fill="#e2e8f0")
        return image

    @staticmethod
    def _timeline_image(segments: list[dict], duration: float, width: int = 1200, height: int = 280) -> Image.Image:
        image = Image.new("RGB", (width, height), "#101827")
        draw = ImageDraw.Draw(image)
        x0, x1 = 80, width - 30
        draw.line((x0, 60, x1, 60), fill="#94a3b8", width=2)
        colors = {"silence": "#64748b", "sound_activity": "#22c55e", "unknown_audio_activity": "#a78bfa",
                  "voiced_activity_candidate": "#38bdf8", "speech": "#f472b6", "potential_acoustic_transition": "#f97316",
                  "synthetic_audio_model_indicator": "#facc15"}
        rows = {"silence": 82, "sound_activity": 112, "unknown_audio_activity": 142,
                "voiced_activity_candidate": 172, "speech": 202, "potential_acoustic_transition": 232,
                "synthetic_audio_model_indicator": 262}
        for segment in segments:
            kind = segment["segment_type"]
            scale = (x1 - x0) / max(duration, 1e-9)
            xs = x0 + segment["start_seconds"] * scale
            xe = x0 + segment["end_seconds"] * scale
            y = rows[kind]
            if xe <= xs:
                draw.line((xs, y - 6, xs, y + 6), fill=colors[kind], width=3)
            else:
                draw.rectangle((xs, y - 5, xe, y + 5), fill=colors[kind])
        draw.text((12, 12), f"Audio timeline · {duration:.3f}s", fill="#e2e8f0")
        for kind, y in rows.items():
            draw.text((12, y - 7), kind.replace("_", " "), fill=colors[kind])
        return image
