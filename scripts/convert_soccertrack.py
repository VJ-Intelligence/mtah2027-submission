"""Convert SoccerTrack v2 game-state JSON (2.7 GB per half) into compact per-half npz files.

Streams the pretty-printed JSON line by line (no full json.load) and keeps, per object
annotation: frame, category, track id, player id, team, jersey, image bbox and the pitch
bottom-middle point (bbox_pitch and bbox_pitch_raw). Pitch-line annotations are dropped.

Verification per half (written to <out>/<half>.verify.json): the number of object
annotations parsed must equal an independent byte-level count of '"supercategory": "object"'
in the source file, and the frame range must match info.seq_length. Only halves that pass
are eligible for raw-JSON deletion (done separately, never by this script).

Usage: python scripts/convert_soccertrack.py <soccertrack-v2 dir> [n_workers]
"""
import json
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

NUM = re.compile(r'^\s*"([A-Za-z_]+)":\s*"?([^",]*)"?,?\s*$')
TEAM = {"left": 0, "right": 1}


def count_objects(path: Path) -> int:
    """Independent count of object annotations: all '"supercategory": "object"' hits minus
    those in the trailing "categories" section (the category definitions)."""
    env = {"LC_ALL": "C"}
    hits = subprocess.run(["grep", "-b", "-o", '"supercategory": "object"', str(path)],
                          capture_output=True, text=True, env=env).stdout.splitlines()
    cat = subprocess.run(["grep", "-b", "-o", '"categories"', str(path)],
                         capture_output=True, text=True, env=env).stdout.splitlines()
    cat_off = int(cat[-1].split(":")[0]) if cat else float("inf")
    return sum(1 for h in hits if int(h.split(":")[0]) < cat_off)


def convert(args):
    src, dst = args
    src, dst = Path(src), Path(dst)
    info, rows = {}, []
    cur, section, in_info, in_ann = None, None, False, False
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not in_ann:
                if s.startswith('"info"'):
                    in_info = True
                    continue
                if in_info:
                    m = NUM.match(line)
                    if m:
                        info[m.group(1)] = m.group(2)
                    elif s.startswith("}"):
                        in_info = False
                    continue
                if s.startswith('"annotations"'):
                    in_ann = True
                continue
            if s.startswith('"categories"'):
                break
            if s.startswith('"id"'):
                if cur is not None and cur.get("supercategory") == "object":
                    rows.append(cur)
                cur, section = {}, None
                continue
            if cur is None:
                continue
            if s.endswith("{"):
                section = s.split('"')[1] if '"' in s else section
                continue
            if s.startswith("}"):
                section = None
                continue
            m = NUM.match(line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            key = f"{section}.{k}" if section else k
            cur[key] = v
    if cur is not None and cur.get("supercategory") == "object":
        rows.append(cur)

    def col(key, dtype, default):
        return np.array([r.get(key, default) if r.get(key, "") not in ("", "null", "None") else default
                         for r in rows], dtype=dtype)

    image_id = np.array([int(r["image_id"]) for r in rows], np.int64)
    frame = (image_id % 1_000_000).astype(np.int32)
    out = {
        "frame": frame,
        "category": col("category_id", np.int8, -1),
        "track_id": col("track_id", np.int32, -1),
        "player_id": col("attributes.player_id", np.int64, -1),
        "team": np.array([TEAM.get(r.get("attributes.team", ""), -1) for r in rows], np.int8),
        "jersey": np.array([r.get("attributes.jersey", "") for r in rows]),
        "role": np.array([r.get("attributes.role", "") for r in rows]),
        "img_xywh": np.stack([col("bbox_image.x", np.float32, np.nan), col("bbox_image.y", np.float32, np.nan),
                              col("bbox_image.w", np.float32, np.nan), col("bbox_image.h", np.float32, np.nan)], 1),
        "pitch_xy": np.stack([col("bbox_pitch.x_bottom_middle", np.float32, np.nan),
                              col("bbox_pitch.y_bottom_middle", np.float32, np.nan)], 1),
        "pitch_raw_xy": np.stack([col("bbox_pitch_raw.x_bottom_middle", np.float32, np.nan),
                                  col("bbox_pitch_raw.y_bottom_middle", np.float32, np.nan)], 1),
        "info_json": np.array(json.dumps(info)),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)

    n_src = count_objects(src)
    seq = int(info.get("seq_length", -1))
    check = {
        "source": src.name,
        "source_bytes": src.stat().st_size,
        "objects_parsed": len(rows),
        "objects_in_source": n_src,
        "seq_length": seq,
        "frame_min": int(frame.min()) if len(frame) else None,
        "frame_max": int(frame.max()) if len(frame) else None,
        "n_frames_with_objects": int(len(np.unique(frame))),
        "categories": {int(c): int((out["category"] == c).sum()) for c in np.unique(out["category"])},
        "nan_pitch_xy": int(np.isnan(out["pitch_xy"]).any(1).sum()),
    }
    check["ok"] = bool(len(rows) == n_src and len(rows) > 0 and check["frame_max"] is not None
                       and 1 <= check["frame_min"] and check["frame_max"] <= seq
                       and check["nan_pitch_xy"] == 0)
    # reload test: the written file must open and match
    with np.load(dst) as z:
        check["ok"] = check["ok"] and len(z["frame"]) == len(rows)
    dst.with_suffix(".verify.json").write_text(json.dumps(check, indent=1))
    return check


if __name__ == "__main__":
    root = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    jobs = [(str(p), str(root / "compact" / (p.stem + ".npz"))) for p in sorted(root.glob("gsr/*/*.json"))]
    with Pool(workers) as pool:
        for c in pool.imap_unordered(convert, jobs):
            print(f"{c['source']}: ok={c['ok']} objects {c['objects_parsed']}/{c['objects_in_source']} "
                  f"frames {c['frame_min']}-{c['frame_max']} of {c['seq_length']} cats {c['categories']}", flush=True)
