"""TruthLens Video Forensics - Multi-Video Comparison Module

Computes perceptual hashes, visual fingerprints, pairwise frame matchings,
similarity matrices, cluster networks, and relationship indicators to prepare
structured comparison features for the TruthLens Correlation Engine.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from truthlens_video.config import (
    PHASH_HAMMING_MATCH_THRESHOLD,
    PHASH_SIZE,
    SHARED_SOURCE_MIN_MATCHING_FRAMES,
    VIDEO_SIMILARITY_MATCH_THRESHOLD,
)
from truthlens_video.frame_extractor import ExtractedFrame


def compute_dhash(image_or_path: Path | str | np.ndarray, hash_size: int = PHASH_SIZE) -> str:
    """Computes a difference hash (dHash) for an image (accepts path or in-memory numpy array)."""
    try:
        if isinstance(image_or_path, np.ndarray):
            if image_or_path.ndim == 3:
                gray = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_or_path
            img = Image.fromarray(gray)
        else:
            img = Image.open(image_or_path).convert("L")

        img = img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(img, dtype=np.int32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return "".join("1" if b else "0" for b in diff.flatten())
    except Exception:
        return "0" * (hash_size * hash_size)


def compute_ahash(image_or_path: Path | str | np.ndarray, hash_size: int = PHASH_SIZE) -> str:
    """Computes an average hash (aHash) for an image (accepts path or in-memory numpy array)."""
    try:
        if isinstance(image_or_path, np.ndarray):
            if image_or_path.ndim == 3:
                gray = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_or_path
            img = Image.fromarray(gray)
        else:
            img = Image.open(image_or_path).convert("L")

        img = img.resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(img, dtype=np.float32)
        avg = np.mean(pixels)
        diff = pixels >= avg
        return "".join("1" if b else "0" for b in diff.flatten())
    except Exception:
        return "0" * (hash_size * hash_size)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculates Hamming distance between two binary hash strings."""
    if len(hash1) != len(hash2):
        return 999
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def compute_color_histogram(image_or_path: Path | str | np.ndarray) -> np.ndarray:
    """Computes normalized HSV 3D color histogram (accepts path or in-memory BGR array)."""
    if isinstance(image_or_path, np.ndarray):
        img = image_or_path
    else:
        img = cv2.imread(str(image_or_path))

    if img is None:
        return np.zeros(32, dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_video_fingerprints(
    evidence_id: str,
    frames: List[ExtractedFrame],
    metadata: Dict[str, Any],
    frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Generates perceptual fingerprints, keyframe hashes, and visual features for a single video.

    Accepts optional in-memory frame_images dict {path: (bgr, gray)} to eliminate redundant disk reads.
    """
    keyframe_hashes: List[Dict[str, Any]] = []
    color_profiles: List[np.ndarray] = []

    for f in frames:
        if frame_images and f.absolute_path in frame_images:
            bgr_img, gray_img = frame_images[f.absolute_path]
            dhash_val = compute_dhash(gray_img)
            ahash_val = compute_ahash(gray_img)
            color_hist = compute_color_histogram(bgr_img)
        else:
            dhash_val = compute_dhash(f.absolute_path)
            ahash_val = compute_ahash(f.absolute_path)
            color_hist = compute_color_histogram(f.absolute_path)

        color_profiles.append(color_hist)

        keyframe_hashes.append({
            "timestamp": f.timestamp_seconds,
            "frame_reference": Path(f.absolute_path).name,
            "dhash": dhash_val,
            "ahash": ahash_val,
        })

    # Sequence average color histogram
    if color_profiles:
        avg_color_hist = np.mean(color_profiles, axis=0).tolist()
    else:
        avg_color_hist = [0.0] * 32

    return {
        "evidence_id": evidence_id,
        "sha256": metadata.get("sha256", ""),
        "duration_seconds": metadata.get("duration_seconds", 0.0),
        "resolution": metadata.get("resolution", {}),
        "total_frames_fingerprinted": len(keyframe_hashes),
        "keyframe_signatures": keyframe_hashes,
        "color_profile_vector": avg_color_hist,
    }


def compare_two_videos(
    vid_a: Dict[str, Any],
    vid_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Compares two videos using their extracted fingerprints and forensic findings."""
    id_a = vid_a["evidence_id"]
    id_b = vid_b["evidence_id"]

    sha_a = vid_a.get("source_matching_features", {}).get("sha256")
    sha_b = vid_b.get("source_matching_features", {}).get("sha256")

    # 1. Exact binary duplicate check
    if sha_a and sha_b and sha_a == sha_b:
        return {
            "target_evidence_id": id_b,
            "comparison_type": "pairwise",
            "relationship": "exact_duplicate",
            "similarity_score": 1.0,
            "matching_frames_count": len(vid_a.get("source_matching_features", {}).get("keyframe_signatures", [])),
            "matched_frame_pairs": [],
            "shared_source_probability": "confirmed_identical_file",
            "notes": "Cryptographic SHA-256 hash match. Files are bitwise identical.",
        }

    # 2. Keyframe perceptual hash alignment
    keys_a = vid_a.get("source_matching_features", {}).get("keyframe_signatures", [])
    keys_b = vid_b.get("source_matching_features", {}).get("keyframe_signatures", [])

    matched_pairs: List[Dict[str, Any]] = []

    for ka in keys_a:
        hash_a = ka.get("dhash", "")
        best_match = None
        min_dist = 999

        for kb in keys_b:
            hash_b = kb.get("dhash", "")
            dist = hamming_distance(hash_a, hash_b)
            if dist < min_dist:
                min_dist = dist
                best_match = kb

        if min_dist <= PHASH_HAMMING_MATCH_THRESHOLD and best_match:
            matched_pairs.append({
                "source_timestamp": ka["timestamp"],
                "target_timestamp": best_match["timestamp"],
                "source_frame": ka["frame_reference"],
                "target_frame": best_match["frame_reference"],
                "hamming_distance": min_dist,
            })

    # Calculate frame match ratio
    min_keys = min(len(keys_a), len(keys_b)) if (keys_a and keys_b) else 1
    match_ratio = len(matched_pairs) / float(min_keys)

    # 3. Color profile correlation
    color_a = np.array(vid_a.get("source_matching_features", {}).get("color_profile_vector", [0]))
    color_b = np.array(vid_b.get("source_matching_features", {}).get("color_profile_vector", [0]))

    if len(color_a) == len(color_b) and len(color_a) > 0 and np.linalg.norm(color_a) > 0 and np.linalg.norm(color_b) > 0:
        color_sim = float(np.dot(color_a, color_b) / (np.linalg.norm(color_a) * np.linalg.norm(color_b)))
        color_sim = max(0.0, min(1.0, color_sim))
    else:
        color_sim = 0.0

    # Composite similarity score
    composite_similarity = round(0.75 * match_ratio + 0.25 * color_sim, 3)
    composite_similarity = min(1.0, max(0.0, composite_similarity))

    # 4. Determine relationship category
    anomalies_a = len(vid_a.get("findings", []))
    anomalies_b = len(vid_b.get("findings", []))

    if len(matched_pairs) >= SHARED_SOURCE_MIN_MATCHING_FRAMES or composite_similarity >= VIDEO_SIMILARITY_MATCH_THRESHOLD:
        if abs(anomalies_a - anomalies_b) >= 2:
            relationship = "potential_manipulation_derivative"
            prob = "probable_derived_or_modified"
            notes = (
                f"Videos share {len(matched_pairs)} visually matching keyframes, but exhibit significant divergence in forensic anomalies "
                f"({anomalies_a} vs {anomalies_b}). One video may be an edited, filtered, or deepfake derivative."
            )
        else:
            relationship = "visually_related_potential_shared_source"
            prob = "possible_common_source"
            notes = (
                f"Videos exhibit high visual similarity ({composite_similarity:.2f}) and share {len(matched_pairs)} matching keyframes. "
                "May share an original capture source or be alternate encodings."
            )
    elif composite_similarity >= 0.30 or len(matched_pairs) >= 1:
        relationship = "partially_related_or_similar_content"
        prob = "inconclusive"
        notes = (
            f"Videos exhibit moderate visual correlation ({composite_similarity:.2f}) with {len(matched_pairs)} isolated frame matches. "
            "May share background elements, themes, or similar lighting."
        )
    else:
        relationship = "unrelated"
        prob = "unlikely"
        notes = "No matching keyframes or significant visual correlation established. Videos appear independent."

    return {
        "target_evidence_id": id_b,
        "comparison_type": "pairwise",
        "relationship": relationship,
        "similarity_score": composite_similarity,
        "matching_frames_count": len(matched_pairs),
        "matched_frame_pairs": matched_pairs[:10],  # Keep top matches
        "shared_source_probability": prob,
        "notes": notes,
    }


def analyze_video_corpus(video_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Performs multi-video corpus comparison across all uploaded videos.
    
    Generates:
        - Pairwise comparison records
        - Similarity matrix
        - Visual relationship clusters
        - Identification of unlinked/unrelated videos
        - Candidate original source identification per cluster
    """
    if not video_results:
        return {"clusters": [], "unlinked_videos": [], "similarity_matrix": {}}

    evidence_ids = [v["evidence_id"] for v in video_results]
    n = len(video_results)

    # Initialize similarity matrix
    similarity_matrix: Dict[str, Dict[str, float]] = {eid: {} for eid in evidence_ids}
    pairwise_relations: List[Dict[str, Any]] = []

    # Graph adjacency for connected components (clustering)
    adjacency: Dict[str, set] = {eid: set() for eid in evidence_ids}

    for i in range(n):
        for j in range(n):
            id_i = evidence_ids[i]
            id_j = evidence_ids[j]
            if i == j:
                similarity_matrix[id_i][id_j] = 1.0
            elif j > i:
                comp = compare_two_videos(video_results[i], video_results[j])
                sim = comp["similarity_score"]
                similarity_matrix[id_i][id_j] = sim
                similarity_matrix[id_j][id_i] = sim

                # Add reciprocal relations to individual video outputs
                rel_for_i = dict(comp)
                rel_for_i["target_evidence_id"] = id_j

                comp_rev = dict(comp)
                comp_rev["target_evidence_id"] = id_i

                # Save into results
                video_results[i].setdefault("video_relationships", []).append(rel_for_i)
                video_results[j].setdefault("video_relationships", []).append(comp_rev)

                pairwise_relations.append({
                    "video_a": id_i,
                    "video_b": id_j,
                    "similarity": sim,
                    "relationship": comp["relationship"],
                    "matching_frames": comp["matching_frames_count"],
                    "notes": comp["notes"],
                })

                if comp["relationship"] in (
                    "exact_duplicate",
                    "potential_manipulation_derivative",
                    "visually_related_potential_shared_source",
                ):
                    adjacency[id_i].add(id_j)
                    adjacency[id_j].add(id_i)

    # Connected component clustering (Breadth-First Search)
    visited = set()
    clusters: List[Dict[str, Any]] = []
    unlinked_videos: List[str] = []

    for eid in evidence_ids:
        if eid not in visited:
            component = []
            queue = [eid]
            visited.add(eid)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adjacency[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) > 1:
                # Determine candidate original source within this cluster
                # Heuristic: Highest resolution and lowest forensic anomaly count
                cluster_videos = [v for v in video_results if v["evidence_id"] in component]
                candidate = min(
                    cluster_videos,
                    key=lambda x: (len(x.get("findings", [])), -x.get("metadata", {}).get("file_size_bytes", 0))
                )

                clusters.append({
                    "cluster_id": f"CLUSTER-{len(clusters) + 1}",
                    "video_count": len(component),
                    "evidence_ids": component,
                    "candidate_original_source": {
                        "evidence_id": candidate["evidence_id"],
                        "file_name": candidate.get("metadata", {}).get("file_name"),
                        "anomaly_count": len(candidate.get("findings", [])),
                        "selection_basis": (
                            "Identified heuristically based on lowest forensic anomaly count and higher file fidelity. "
                            "Not verified ground-truth."
                        ),
                    },
                })
            else:
                unlinked_videos.append(eid)

    return {
        "total_videos": n,
        "clusters": clusters,
        "unlinked_videos": unlinked_videos,
        "pairwise_comparisons": pairwise_relations,
        "similarity_matrix": similarity_matrix,
    }
