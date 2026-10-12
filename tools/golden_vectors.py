import json
import os
from typing import List, Dict, Any
from tools.motion_presets import preset_to_camera, PRESETS

def generate_golden_cases() -> List[Dict[str, Any]]:
    cases = []
    
    presets_list = ["push_in_gentle", "push_in_strong", "push_slide", "slide", "hero_sweep", "tilt", "pull_back"]
    aspects = {
        "3:2": (1500, 1000),
        "4:3": (1600, 1200),
        "9:16": (1080, 1920)
    }
    sx_list = [0.3, 0.5, 0.7]
    order_list = [0, 1]

    # Systematic combination across presets, aspects, subject_x, and order parity
    # 5 presets x 3 aspects = 15 core combinations with alternating sx and order
    idx = 0
    for p in presets_list:
        for asp_name, (sw, sh) in aspects.items():
            sx = sx_list[idx % len(sx_list)]
            order = order_list[idx % len(order_list)]
            cases.append({
                "preset": p,
                "src_w": sw,
                "src_h": sh,
                "out_w": 1920,
                "out_h": 1080,
                "subject_x": sx,
                "subject_y": 0.5,
                "order": order,
                "max_zoom": 1.6
            })
            idx += 1

    # Add complementary cases to ensure every (preset, sx, order) combination has direct coverage
    for p in ["push_slide", "slide"]:
        for sx in [0.3, 0.7]:
            for order in [0, 1]:
                cases.append({
                    "preset": p,
                    "src_w": 1500,
                    "src_h": 1000,
                    "out_w": 1920,
                    "out_h": 1080,
                    "subject_x": sx,
                    "subject_y": 0.5,
                    "order": order,
                    "max_zoom": 1.6
                })

    for p in ["push_in_gentle", "pull_back"]:
        cases.append({
            "preset": p,
            "src_w": 1600,
            "src_h": 1200,
            "out_w": 1920,
            "out_h": 1080,
            "subject_x": 0.5,
            "subject_y": 0.5,
            "order": 1,
            "max_zoom": 1.6
        })

    # Cases explicitly triggering each fallback branch:
    # 1. scale_0.75
    cases.append({
        "preset": "push_slide",
        "src_w": 1080,
        "src_h": 1920,
        "out_w": 1920,
        "out_h": 1080,
        "subject_x": 0.5,
        "subject_y": 0.5,
        "order": 0,
        "max_zoom": 1.10
    })
    # 2. scale_0.5
    cases.append({
        "preset": "push_slide",
        "src_w": 1080,
        "src_h": 1920,
        "out_w": 1920,
        "out_h": 1080,
        "subject_x": 0.5,
        "subject_y": 0.5,
        "order": 0,
        "max_zoom": 1.06
    })
    # 3. downgrade_push_in_gentle
    cases.append({
        "preset": "push_slide",
        "src_w": 1080,
        "src_h": 1920,
        "out_w": 1920,
        "out_h": 1080,
        "subject_x": 0.5,
        "subject_y": 0.5,
        "order": 0,
        "max_zoom": 1.00
    })
    # 4. static
    cases.append({
        "preset": "slide",
        "src_w": 1080,
        "src_h": 1920,
        "out_w": 1920,
        "out_h": 1080,
        "subject_x": 0.5,
        "subject_y": 0.5,
        "order": 0,
        "max_zoom": 0.99
    })

    # De-duplicate while preserving order
    unique_cases = []
    seen = set()
    for c in cases:
        key = (c["preset"], c["src_w"], c["src_h"], c["subject_x"], c["order"], c["max_zoom"])
        if key not in seen:
            seen.add(key)
            unique_cases.append(c)

    return unique_cases

def generate_golden_vectors() -> List[Dict[str, Any]]:
    cases = generate_golden_cases()
    results = []
    for c in cases:
        cam, fb = preset_to_camera(
            preset=c["preset"],
            src_w=c["src_w"],
            src_h=c["src_h"],
            out_w=c["out_w"],
            out_h=c["out_h"],
            subject_x=c["subject_x"],
            subject_y=c["subject_y"],
            order=c["order"],
            max_zoom=c["max_zoom"]
        )
        entry = {
            "inputs": {
                "preset": c["preset"],
                "src_w": c["src_w"],
                "src_h": c["src_h"],
                "out_w": c["out_w"],
                "out_h": c["out_h"],
                "subject_x": c["subject_x"],
                "subject_y": c["subject_y"],
                "order": c["order"],
                "max_zoom": c["max_zoom"]
            },
            "camera": cam,
            "fallback": fb
        }
        results.append(entry)
    return results

def main():
    vectors = generate_golden_vectors()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden_vectors.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=2)
    print(f"Generated {len(vectors)} golden vectors in {out_path}")

if __name__ == "__main__":
    main()
