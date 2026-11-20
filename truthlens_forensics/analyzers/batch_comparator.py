"""
Multi-Image Batch Comparative Forensics Module
Performs cross-image sensor PRNU correlation, metadata timeline comparison,
and categorizes all submitted evidence items in a batch into authentic, deepfake/manipulated,
AI-generated, or inconclusive candidates.
"""
import numpy as np
from typing import List, Dict, Any


class BatchComparator:
    def compare_batch(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        authentic_candidates = []
        manipulated_candidates = []
        ai_generated_candidates = []
        inconclusive_candidates = []

        # 1. Categorize per-image based on overall_forensic_assessment
        for report in reports:
            ev_id = report.get("evidence_id", "UNKNOWN")
            analyses = report.get("analyses", {})
            overall = analyses.get("overall_forensic_assessment", {})
            cat = overall.get("category", "inconclusive")

            if cat == "likely_original_capture":
                authentic_candidates.append(ev_id)
            elif cat == "likely_deepfake_or_manipulated":
                manipulated_candidates.append(ev_id)
            elif cat == "likely_ai_generated":
                ai_generated_candidates.append(ev_id)
            else:
                inconclusive_candidates.append(ev_id)

        # 2. Cross-image PRNU sensor noise correlation & metadata comparison
        correlations = []
        n = len(reports)
        for i in range(n):
            for j in range(i + 1, n):
                rep_a = reports[i]
                rep_b = reports[j]
                id_a = rep_a.get("evidence_id", f"EVD-{i+1}")
                id_b = rep_b.get("evidence_id", f"EVD-{j+1}")

                noise_a = rep_a.get("analyses", {}).get("noise_residual_analysis", {})
                noise_b = rep_b.get("analyses", {}).get("noise_residual_analysis", {})

                meta_a = rep_a.get("analyses", {}).get("metadata_analysis", {})
                meta_b = rep_b.get("analyses", {}).get("metadata_analysis", {})

                cam_a = meta_a.get("camera_info", {})
                cam_b = meta_b.get("camera_info", {})

                # Compute cosine similarity between PRNU normalized fingerprint vectors
                vec_a = noise_a.get("prnu_fingerprint_vector", [])
                vec_b = noise_b.get("prnu_fingerprint_vector", [])

                prnu_corr = 0.0
                if vec_a and vec_b and len(vec_a) == len(vec_b):
                    va = np.array(vec_a, dtype=np.float32)
                    vb = np.array(vec_b, dtype=np.float32)
                    norm_a = np.linalg.norm(va) + 1e-8
                    norm_b = np.linalg.norm(vb) + 1e-8
                    prnu_corr = float(np.dot(va, vb) / (norm_a * norm_b))
                    prnu_corr = round(max(min(prnu_corr, 1.0), -1.0), 4)

                shared_camera = False
                make_a, mod_a = cam_a.get("make"), cam_a.get("model")
                make_b, mod_b = cam_b.get("make"), cam_b.get("model")
                if make_a and make_b and make_a == make_b and mod_a == mod_b:
                    shared_camera = True

                # Relationship assessment
                if prnu_corr >= 0.65 or (shared_camera and prnu_corr >= 0.40):
                    assessment = (
                        f"High sensor correlation ({prnu_corr:.2f}) indicates both images originate from the exact same camera sensor."
                        + (" Shared hardware EXIF tags confirmed." if shared_camera else "")
                    )
                elif shared_camera:
                    assessment = (
                        f"Shared camera model tags ({make_a} {mod_a}), but divergence in high-frequency PRNU ({prnu_corr:.2f}) "
                        "suggests distinct sensor chips, heavy recompression, or digital synthesis."
                    )
                elif prnu_corr < 0.20 and (id_a in ai_generated_candidates or id_b in ai_generated_candidates):
                    assessment = (
                        f"Uncorrelated sensor noise ({prnu_corr:.2f}). One or both images exhibit synthetic/AI origin with no physical sensor PRNU."
                    )
                else:
                    assessment = (
                        f"Low sensor noise correlation ({prnu_corr:.2f}) consistent with independent physical capture devices or unlinked sources."
                    )

                correlations.append({
                    "evidence_pair": [id_a, id_b],
                    "prnu_sensor_correlation": prnu_corr,
                    "shared_camera_model": shared_camera,
                    "relationship_assessment": assessment,
                })

        # 3. Construct summary
        summary_parts = []
        if authentic_candidates:
            summary_parts.append(f"{len(authentic_candidates)} candidate(s) verified as authentic original capture ({', '.join(authentic_candidates)})")
        if manipulated_candidates:
            summary_parts.append(f"{len(manipulated_candidates)} candidate(s) flagged for localized manipulation/deepfake alteration ({', '.join(manipulated_candidates)})")
        if ai_generated_candidates:
            summary_parts.append(f"{len(ai_generated_candidates)} candidate(s) categorized as synthetic AI generation ({', '.join(ai_generated_candidates)})")
        if inconclusive_candidates:
            summary_parts.append(f"{len(inconclusive_candidates)} candidate(s) evaluated as inconclusive ({', '.join(inconclusive_candidates)})")

        summary_text = (
            "Multi-image batch comparative forensic analysis completed across "
            f"{len(reports)} submitted items: " + "; ".join(summary_parts) + "."
        )

        return {
            "authentic_original_candidates": authentic_candidates,
            "deepfake_or_manipulated_candidates": manipulated_candidates,
            "ai_generated_candidates": ai_generated_candidates,
            "inconclusive_candidates": inconclusive_candidates,
            "cross_image_correlations": correlations,
            "summary": summary_text,
        }

