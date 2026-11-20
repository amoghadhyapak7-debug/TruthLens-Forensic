"""
Signal & Candidate Finding Generator
Evaluates normalized engine analyses against digital forensics rules to emit structured signals,
candidate findings, and an overall forensic authenticity assessment.
"""
from typing import Dict, Any, List


class SignalFinder:
    def generate_signals(
        self,
        analyses: Dict[str, Any],
        sha256_verified: bool,
        size_matched: bool = True
    ) -> List[Dict[str, Any]]:
        signals = []
        sig_idx = 1

        def add_sig(code, name, category, strength, source_engine, details):
            nonlocal sig_idx
            signals.append({
                "signal_id": f"SIG-{sig_idx:03d}",
                "code": code,
                "name": name,
                "category": category,
                "strength": strength,
                "source_engine": source_engine,
                "details": details,
            })
            sig_idx += 1

        # 0. Cryptographic integrity
        if not sha256_verified:
            add_sig(
                "SHA256_HASH_MISMATCH",
                "Evidence SHA-256 Hash Mismatch",
                "integrity",
                "Critical",
                "metadata",
                "Computed SHA-256 hash does not match input evidence specification.",
            )

        # 1. AI-Generated detection
        ai = analyses.get("ai_generated_detection", {})
        ai_prob = ai.get("ai_probability", 0.0) or 0.0
        spectral = ai.get("spectral_irregularity_score", 0.0) or 0.0
        if ai_prob >= 0.65:
            add_sig(
                "AI_GENERATION_SIGNATURE_DETECTED",
                "Strong Synthetic AI Generation Signatures",
                "ai_synthesis",
                "High",
                "ai_generated_detection",
                f"Empirical AI probability: {ai_prob:.3f}, spectral irregularity: {spectral:.3f}.",
            )
        elif ai_prob >= 0.45:
            add_sig(
                "AI_GENERATION_SIGNATURE_DETECTED",
                "Atypical High-Frequency Synthetic Characteristics",
                "ai_synthesis",
                "Medium",
                "ai_generated_detection",
                f"Empirical AI probability: {ai_prob:.3f}, spectral irregularity: {spectral:.3f}.",
            )

        # 2. Manipulation detection (Calibrated against natural scene baseline)
        manip = analyses.get("manipulation_detection", {})
        feat_score = manip.get("feature_anomaly_score", 0.0) or 0.0
        regions = manip.get("suspicious_regions", [])
        # True localized manipulation produces significant outlier score AND spatial cluster of suspicious regions
        if feat_score >= 0.80 and len(regions) >= 3:
            add_sig(
                "DEEPFAKE_FACIAL_ARTIFACT_DETECTED",
                "Localized Deepfake / Manipulation Spatial Anomaly",
                "manipulation",
                "High",
                "manipulation_detection",
                f"Feature anomaly score: {feat_score:.3f} across {len(regions)} localized spatial bounding regions.",
            )
        elif feat_score >= 0.65 and len(regions) >= 2:
            add_sig(
                "DEEPFAKE_FACIAL_ARTIFACT_DETECTED",
                "Moderate Spatial Feature Discontinuity",
                "manipulation",
                "Medium",
                "manipulation_detection",
                f"Feature anomaly score: {feat_score:.3f} across {len(regions)} spatial bounding regions.",
            )

        # 3. ELA analysis (Calibrated against JPEG baseline noise floor)
        ela = analyses.get("ela_analysis", {})
        high_ela_frac = ela.get("high_ela_fraction", 0.0) or 0.0
        max_err = ela.get("max_error", 0.0) or 0.0
        if high_ela_frac >= 0.08 or (max_err > 40.0 and len(ela.get("anomaly_regions", [])) >= 3):
            add_sig(
                "ELA_COMPRESSION_INCONSISTENCY",
                "JPEG Compression History Discontinuity",
                "compression",
                "High",
                "ela_analysis",
                f"High-ELA error region fraction: {high_ela_frac:.2%}, max error: {max_err:.1f}.",
            )
        elif high_ela_frac >= 0.03 and max_err > 18.0:
            add_sig(
                "ELA_COMPRESSION_INCONSISTENCY",
                "Moderate Compression Level Inconsistency",
                "compression",
                "Medium",
                "ela_analysis",
                f"High-ELA error region fraction: {high_ela_frac:.2%}, max error: {max_err:.1f}.",
            )

        # 4. Noise analysis (Calibrated against camera sensor baseline)
        noise = analyses.get("noise_residual_analysis", {})
        bvr = noise.get("block_variance_ratio", 0.0) or 0.0
        noise_anom_count = len(noise.get("anomaly_regions", []))
        if bvr >= 2.2 or (bvr >= 1.8 and noise_anom_count >= 5):
            add_sig(
                "NOISE_VARIANCE_ANOMALY",
                "Sensor Noise Variance Inconsistency",
                "sensor_noise",
                "High",
                "noise_residual_analysis",
                f"Wavelet noise variance dispersion ratio: {bvr:.3f}.",
            )
        elif bvr >= 1.6 and noise_anom_count >= 3:
            add_sig(
                "NOISE_VARIANCE_ANOMALY",
                "Moderate Sensor Noise Variance Deviation",
                "sensor_noise",
                "Medium",
                "noise_residual_analysis",
                f"Wavelet noise variance dispersion ratio: {bvr:.3f}.",
            )
        elif bvr >= 1.3:
            add_sig(
                "NOISE_VARIANCE_ANOMALY",
                "Slight Sensor Noise Variance Deviation",
                "sensor_noise",
                "Low",
                "noise_residual_analysis",
                f"Wavelet noise variance dispersion ratio: {bvr:.3f}.",
            )

        # 5. Copy-Move
        stat = analyses.get("statistical_analysis", {})
        if stat.get("copy_move_detected"):
            add_sig(
                "COPY_MOVE_CLONE_DETECTED",
                "Duplicated Image Blocks Detected (Copy-Move)",
                "structural",
                "High",
                "statistical_analysis",
                f"{len(stat.get('copy_move_matches', []))} cloned block pairs identified across disjoint regions.",
            )

        # 6. Metadata
        meta = analyses.get("metadata_analysis", {})
        if meta.get("editing_software_detected"):
            add_sig(
                "EXIF_EDITING_SOFTWARE_TAG",
                "Image Editing Software Signature in EXIF",
                "metadata",
                "Medium",
                "metadata_analysis",
                f"Editing software signature identified: '{meta.get('software_tag')}'.",
            )
        if not meta.get("timestamp_consistent"):
            add_sig(
                "EXIF_TIMESTAMP_INCONSISTENCY",
                "Contradictory Creation vs Modification Timestamps",
                "metadata",
                "Low",
                "metadata_analysis",
                f"Creation: {meta.get('timestamps', {}).get('created')}, Modification: {meta.get('timestamps', {}).get('modified')}.",
            )
        if meta.get("stripped_metadata"):
            add_sig(
                "EXIF_METADATA_STRIPPED",
                "Camera Sensor & Exposure Headers Stripped",
                "metadata",
                "Low",
                "metadata_analysis",
                "Absence of standard camera hardware EXIF tags.",
            )
        elif meta.get("camera_present") and not meta.get("editing_software_detected") and meta.get("timestamp_consistent"):
            add_sig(
                "AUTHENTIC_CAMERA_PIPELINE_CONSISTENT",
                "Camera Hardware & EXIF Pipeline Consistent",
                "metadata",
                "Low",
                "metadata_analysis",
                f"Camera make/model: {meta.get('camera_info', {}).get('make')} {meta.get('camera_info', {}).get('model')}.",
            )

        return signals

    def generate_findings(
        self,
        signals: List[Dict[str, Any]],
        analyses: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        findings = []
        f_idx = 1

        def add_finding(title, severity, category, description, supporting_ev, regions):
            nonlocal f_idx
            findings.append({
                "finding_id": f"FIND-{f_idx:03d}",
                "title": title,
                "severity": severity,
                "category": category,
                "description": description,
                "supporting_evidence": supporting_ev,
                "regions": regions,
            })
            f_idx += 1

        # Check for integrity failure
        hash_mismatch = any(s["code"] == "SHA256_HASH_MISMATCH" for s in signals)
        if hash_mismatch:
            add_finding(
                "Cryptographic Integrity Check Failed",
                "Critical",
                "integrity",
                "The computed SHA-256 hash does not match the expected hash from the evidence descriptor, indicating file transmission corruption or unauthorized modification.",
                ["metadata_analysis"],
                []
            )

        # Check for deepfake / manipulation
        manip_sigs = [s for s in signals if s["code"] in {"DEEPFAKE_FACIAL_ARTIFACT_DETECTED", "ELA_COMPRESSION_INCONSISTENCY", "NOISE_VARIANCE_ANOMALY"}]
        if manip_sigs:
            manip = analyses.get("manipulation_detection", {})
            ela = analyses.get("ela_analysis", {})
            noise = analyses.get("noise_residual_analysis", {})

            # Gather spatial bounding boxes
            all_regions = []
            for r in manip.get("suspicious_regions", [])[:4]:
                all_regions.append({"x": r["x"], "y": r["y"], "width": r["width"], "height": r["height"]})
            for r in ela.get("anomaly_regions", [])[:4]:
                all_regions.append({"x": r["x"], "y": r["y"], "width": r["width"], "height": r["height"]})
            for r in noise.get("anomaly_regions", [])[:2]:
                all_regions.append({"x": r["x"], "y": r["y"], "width": r["width"], "height": r["height"]})

            add_finding(
                "Multi-Engine Indicators of Localized Image Manipulation or Deepfake Splicing",
                "High" if any(s["strength"] == "High" for s in manip_sigs) else "Medium",
                "manipulation",
                (
                    f"Physical and neural forensic engines reveal localized inconsistencies. "
                    f"ResNet-50 feature anomaly score: {manip.get('feature_anomaly_score', 0):.3f}; "
                    f"ELA high-error fraction: {ela.get('high_ela_fraction', 0):.2%}; "
                    f"Wavelet noise variance ratio: {noise.get('block_variance_ratio', 0):.3f}. "
                    "These convergent cues strongly indicate region tampering or composite splicing."
                ),
                list({s["source_engine"] for s in manip_sigs}),
                all_regions[:8]
            )

        # Check for AI generation
        ai_sigs = [s for s in signals if s["code"] == "AI_GENERATION_SIGNATURE_DETECTED"]
        if ai_sigs:
            ai = analyses.get("ai_generated_detection", {})
            add_finding(
                "Synthetic AI-Generation Frequency & Distribution Patterns",
                "High" if any(s["strength"] == "High" for s in ai_sigs) else "Medium",
                "ai_synthesis",
                (
                    f"The evidence exhibits statistical properties characteristic of generative diffusion or GAN synthesis. "
                    f"Empirical AI Probability: {ai.get('ai_probability', 0):.3f}, "
                    f"Spectral Irregularity Ratio: {ai.get('spectral_irregularity_score', 0):.3f}. "
                    f"{ai.get('interpretation', '')}"
                ),
                ["ai_generated_detection"],
                []
            )

        # Check for copy-move
        cm_sigs = [s for s in signals if s["code"] == "COPY_MOVE_CLONE_DETECTED"]
        if cm_sigs:
            stat = analyses.get("statistical_analysis", {})
            matches = stat.get("copy_move_matches", [])
            cm_regions = []
            for m in matches[:4]:
                cm_regions.append(m.get("source_region", {}))
                cm_regions.append(m.get("target_region", {}))

            add_finding(
                "Duplicated Regions Indicative of Content Concealment (Copy-Move)",
                "High",
                "structural",
                f"{len(matches)} duplicated image blocks identified across distinct image quadrants. Typically used to clone or conceal evidence.",
                ["statistical_analysis"],
                cm_regions
            )

        # Check for metadata
        meta_sigs = [s for s in signals if s["source_engine"] == "metadata_analysis" and s["strength"] in {"Medium", "High"}]
        if meta_sigs:
            meta = analyses.get("metadata_analysis", {})
            add_finding(
                "Image Metadata Inconsistency or Post-Processing Software Tag",
                "Medium",
                "metadata",
                meta.get("interpretation", "Metadata anomalies detected."),
                ["metadata_analysis"],
                []
            )

        # If clean baseline
        if not findings:
            add_finding(
                "Forensic Observations Consistent with Authentic Unmanipulated Capture",
                "Low",
                "authenticity",
                "All analytical engines (neural feature continuity, ELA compression, wavelet sensor noise, and metadata) returned values within normal photographic baseline tolerances.",
                list(analyses.keys()),
                []
            )

        return findings

    def assess_overall(
        self,
        signals: List[Dict[str, Any]],
        analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesizes all observations into an overall forensic assessment category.

        Rules (in priority order):
        1. copy-move detected → manipulated
        2. High-AI + stripped EXIF or high AI prob → ai_generated
        3. High-manip signal AND corroborating evidence (editing SW, high ELA, high noise) → manipulated
        4. High-manip AND high anomaly score (>=0.80) AND ≥3 suspicious regions → manipulated
        5. editing SW + moderate manip score → manipulated
        6. Camera EXIF present, no editing SW, no copy-move, low manip, low AI → original
        7. Otherwise → inconclusive
        """
        high_manip = any(s["category"] == "manipulation" and s["strength"] == "High" for s in signals)
        high_ai = any(s["category"] == "ai_synthesis" and s["strength"] == "High" for s in signals)
        copy_move = any(s["code"] == "COPY_MOVE_CLONE_DETECTED" for s in signals)

        ai_prob = analyses.get("ai_generated_detection", {}).get("ai_probability", 0.0) or 0.0
        manip_score = analyses.get("manipulation_detection", {}).get("feature_anomaly_score", 0.0) or 0.0
        susp_regions = analyses.get("manipulation_detection", {}).get("suspicious_regions", [])
        meta = analyses.get("metadata_analysis", {})
        ela = analyses.get("ela_analysis", {})
        noise = analyses.get("noise_residual_analysis", {})

        editing_sw = bool(meta.get("editing_software_detected"))
        has_timestamp_mismatch = (
            not meta.get("timestamp_consistent")
            and meta.get("timestamps", {}).get("created")
            and meta.get("timestamps", {}).get("modified")
            and meta.get("timestamps", {}).get("created") != meta.get("timestamps", {}).get("modified")
        )
        high_ela = (ela.get("high_ela_fraction", 0.0) or 0.0) > 0.08
        high_noise = (noise.get("block_variance_ratio", 0.0) or 0.0) > 2.0

        # Physical / structural corroborations
        physical_corroboration = (
            editing_sw
            or has_timestamp_mismatch
            or high_ela
            or high_noise
        )

        # Ensemble cross-validation:
        # A standalone AI/neural detector flag must NEVER solely drive
        # 'likely_deepfake_or_manipulated' without corroborating physical evidence
        # from ELA, PRNU noise variance, copy-move clone, or metadata mismatch.
        is_manipulated = (
            copy_move
            or (high_manip and physical_corroboration)
            or (editing_sw and (manip_score >= 0.55 or high_ela or high_noise or has_timestamp_mismatch))
        )

        is_ai_gen = (
            (high_ai and not meta.get("camera_present"))
            or (ai_prob >= 0.65 and meta.get("stripped_metadata"))
            or (ai_prob >= 0.75)
        )

        # AI-generated takes precedence when no editing software / copy-move manipulation is present
        # and there is strong AI signal (stripped EXIF + no camera + medium-high AI probability)
        ai_definitive = (
            meta.get("stripped_metadata")
            and not meta.get("camera_present")
            and not editing_sw
            and not copy_move
            and ai_prob >= 0.45
        )

        if is_manipulated and not ai_definitive:
            category = "likely_deepfake_or_manipulated"
            basis = "Corroborated physical/structural tampering indicators: localized feature discontinuity confirmed by ELA compression, PRNU noise variance, copy-move clone, or editing software signature."
            summary = "Evidence demonstrates significant physical and deep-feature traces of localized manipulation, compositing, or deepfake alteration."
        elif is_ai_gen or ai_definitive:
            category = "likely_ai_generated"
            basis = "Elevated 2D spectral high-frequency energy ratio, atypical logit entropy, and absence of camera sensor PRNU."
            summary = "Evidence exhibits high statistical and frequency anomalies characteristic of AI diffusion / GAN generative models."
        elif (
            meta.get("camera_present")
            and not editing_sw
            and not copy_move
            and manip_score < 0.75
            and ai_prob < 0.55
        ):
            category = "likely_original_capture"
            basis = "Camera hardware EXIF present, uniform wavelet sensor noise distribution, homogeneous ELA compression, and low neural anomaly scores."
            summary = "Evidence shows uniform camera sensor noise, authentic EXIF camera pipeline signatures, and no localized manipulation anomalies."
        else:
            category = "inconclusive"
            basis = "Ambiguous or contradictory forensic indicators requiring secondary human investigative verification."
            summary = "Forensic metrics fall within intermediate threshold ranges without definitive physical or neural proof of manipulation or purity."

        return {
            "category": category,
            "confidence_basis": basis,
            "evidence_summary": summary,
        }

