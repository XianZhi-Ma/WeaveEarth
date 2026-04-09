import argparse
import hashlib
import json
import logging
import math
import os
import re
import string
import time
import warnings
from collections import defaultdict
from typing import Dict

import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

from qwen_vl_utils import process_vision_info

try:
    from loguru import logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    class _CompatLogger:
        def __init__(self):
            self._logger = logging.getLogger("eval_lrs_vqa_qwen3vl_routed")

        def info(self, msg, *args):
            self._logger.info(msg.format(*args))

        def warning(self, msg, *args):
            self._logger.warning(msg.format(*args))

        def error(self, msg, *args):
            self._logger.error(msg.format(*args))

    logger = _CompatLogger()


PILImage.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_image_index(dataset_root):
    index = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    for root, _, files in os.walk(dataset_root):
        for name in files:
            if name.lower().endswith(valid_exts):
                index[name] = os.path.join(root, name)
    return index


def resolve_image_path(image_rel_path, dataset_root, image_index):
    direct_path = os.path.join(dataset_root, image_rel_path)
    if os.path.exists(direct_path):
        return direct_path
    base = os.path.basename(image_rel_path)
    return image_index.get(base, "")


def keep_item(item, args):
    category = str(item.get("category", ""))
    size_bin = str(item.get("size_bin", ""))
    if args.filter_category and category not in {x.strip() for x in args.filter_category.split(",") if x.strip()}:
        return False
    if args.filter_size_bin and size_bin not in {x.strip() for x in args.filter_size_bin.split(",") if x.strip()}:
        return False
    return True


def normalize_text(text):
    text = str(text).strip().lower().replace("\n", " ").replace("\t", " ")
    text = re.split(r"(?:\bbecause\b|\btherefore\b|\bso\b|\.|;)", text)[0]
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    yn_map = {"y": "yes", "yeah": "yes", "true": "yes", "1": "yes", "n": "no", "false": "no", "0": "no"}
    num_word_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    if text in yn_map:
        text = yn_map[text]
    if text in num_word_map:
        text = num_word_map[text]
    return text


def extract_model_answer(raw_text):
    text = raw_text.strip()
    answer_tag = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip(), "answer_tag"
    for candidate in [text, raw_text]:
        json_match = re.search(r"\{[\s\S]*?\}", candidate)
        if not json_match:
            continue
        try:
            obj = json.loads(json_match.group(0))
            for k in ["response", "answer", "final_answer", "prediction"]:
                if k in obj:
                    return str(obj[k]).strip(), "json"
        except Exception:
            pass
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return (lines[0], "first_line") if lines else ("", "empty")


def token_f1(pred, gt):
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    gt_cnt = defaultdict(int)
    for t in gt_tokens:
        gt_cnt[t] += 1
    overlap = 0
    for t in pred_tokens:
        if gt_cnt[t] > 0:
            overlap += 1
            gt_cnt[t] -= 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_match(pred_text, gt_text):
    pred = normalize_text(pred_text)
    gt = normalize_text(gt_text)
    if not pred or not gt:
        return False, pred, gt, "empty"
    if pred == gt:
        return True, pred, gt, "exact"
    if pred.isdigit() and gt.isdigit() and int(pred) == int(gt):
        return True, pred, gt, "int_exact"
    if len(pred) >= 3 and len(gt) >= 3 and (pred in gt or gt in pred):
        return True, pred, gt, "substring"
    if token_f1(pred, gt) >= 0.8:
        return True, pred, gt, "token_f1"
    return False, pred, gt, "mismatch"


def group_acc(records):
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        grouped[r["group"]]["total"] += 1
        if r["is_correct"]:
            grouped[r["group"]]["correct"] += 1
    out = {}
    for k, v in grouped.items():
        total = v["total"]
        out[k] = {"correct": v["correct"], "total": total, "acc": (v["correct"] / total if total > 0 else 0.0)}
    return out


def resize_long_side(image, target_long_side):
    w, h = image.size
    if target_long_side <= 0:
        return image
    if max(w, h) <= target_long_side:
        return image
    scale = target_long_side / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), PILImage.BILINEAR)


def sanitize_rel_pos(text):
    return str(text).replace("_", " ")


def relative_position_from_indices(row, col, rows, cols):
    if row <= rows // 3:
        v = "top"
    elif row >= (2 * rows) // 3:
        v = "bottom"
    else:
        v = "middle"
    if col <= cols // 3:
        h = "left"
    elif col >= (2 * cols) // 3:
        h = "right"
    else:
        h = "center"
    return f"{v}_{h}"


def make_grid_regions(width, height, rows, cols, overlap_ratio):
    regions = []
    tile_w = width / cols
    tile_h = height / rows
    overlap_w = tile_w * overlap_ratio
    overlap_h = tile_h * overlap_ratio
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(round(c * tile_w - overlap_w / 2)))
            y1 = max(0, int(round(r * tile_h - overlap_h / 2)))
            x2 = min(width, int(round((c + 1) * tile_w + overlap_w / 2)))
            y2 = min(height, int(round((r + 1) * tile_h + overlap_h / 2)))
            idx = r * cols + c + 1
            regions.append(
                {
                    "id": f"R{idx}",
                    "row": r,
                    "col": c,
                    "bbox": (x1, y1, x2, y2),
                    "relative_position": relative_position_from_indices(r, c, rows, cols),
                }
            )
    return regions


def crop_regions(image, regions):
    out = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        item = dict(region)
        item["image"] = image.crop((x1, y1, x2, y2)).convert("RGB")
        out.append(item)
    return out


def draw_labeled_grid(image, rows, cols):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    w, h = out.size
    for r in range(rows + 1):
        y = int(round(r * h / rows))
        draw.line((0, y, w, y), fill=(255, 0, 0), width=2)
    for c in range(cols + 1):
        x = int(round(c * w / cols))
        draw.line((x, 0, x, h), fill=(255, 0, 0), width=2)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1
            x = int(round(c * w / cols)) + 4
            y = int(round(r * h / rows)) + 4
            draw.rectangle((x - 2, y - 2, x + 40, y + 14), fill=(0, 0, 0))
            draw.text((x, y), f"R{idx}", fill=(255, 255, 255), font=font)
    return out


def build_region_metadata_text(regions):
    lines = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        lines.append(f"{region['id']}: row={region['row']}, col={region['col']}, pos={region['relative_position']}, bbox=[{x1},{y1},{x2},{y2}]")
    return "\n".join(lines)


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    denom = area1 + area2 - inter
    return inter / denom if denom > 0 else 0.0


def enrich_region_metadata(regions, rows, cols, image_w, image_h, anchor_ids=None, support_ids=None):
    anchor_ids = set(anchor_ids or [])
    support_ids = set(support_ids or [])
    enriched = []
    for region in regions:
        item = dict(region)
        x1, y1, x2, y2 = item["bbox"]
        cx = ((x1 + x2) / 2.0) / max(1.0, float(image_w))
        cy = ((y1 + y2) / 2.0) / max(1.0, float(image_h))
        bw = (x2 - x1) / max(1.0, float(image_w))
        bh = (y2 - y1) / max(1.0, float(image_h))
        item["global_box_norm"] = [cx, cy, bw, bh]
        neighbor_ids = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = item["row"] + dr
                nc = item["col"] + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                neighbor_ids.append(f"R{nr * cols + nc + 1}")
        item["neighbor_ids"] = neighbor_ids
        if item["id"] in anchor_ids:
            role = "anchor"
        elif item["id"] in support_ids:
            role = "support"
        else:
            role = "context"
        item["role"] = role
        item["is_anchor"] = item["id"] in anchor_ids
        item["neighbor_distance"] = 0 if item["is_anchor"] else 1
        item["scale_token"] = "grid"
        enriched.append(item)
    return enriched


def select_minimal_support_set(candidate_regions, support_set_budget, support_iou_threshold, semantic_score_weight=1.0, anchor_bonus=0.08):
    if not candidate_regions:
        return []
    budget = max(1, int(support_set_budget))
    anchors = [r for r in candidate_regions if r.get("is_anchor", False)]
    anchors.sort(key=lambda r: (r.get("router_rank", 10**9), r["row"], r["col"]))
    selected = anchors[:budget]
    selected_ids = {r["id"] for r in selected}
    selected_rows = {r["row"] for r in selected}
    selected_cols = {r["col"] for r in selected}
    while len(selected) < budget:
        best_region = None
        best_key = None
        for region in candidate_regions:
            if region["id"] in selected_ids:
                continue
            max_iou = max((bbox_iou(region["bbox"], s["bbox"]) for s in selected), default=0.0)
            new_row_gain = 1 if region["row"] not in selected_rows else 0
            new_col_gain = 1 if region["col"] not in selected_cols else 0
            overlap_penalty = 1 if max_iou >= support_iou_threshold else 0
            semantic_score = float(region.get("semantic_score", 0.0))
            anchor_term = anchor_bonus if region.get("is_anchor", False) else 0.0
            score = semantic_score_weight * semantic_score + anchor_term + 2.5 * (new_row_gain + new_col_gain) - 2.0 * overlap_penalty - max_iou
            key = (score, semantic_score, -max_iou, -region["neighbor_distance"], -region["row"], -region["col"])
            if best_key is None or key > best_key:
                best_key = key
                best_region = region
        if best_region is None:
            break
        selected.append(best_region)
        selected_ids.add(best_region["id"])
        selected_rows.add(best_region["row"])
        selected_cols.add(best_region["col"])
        if len(selected_ids) >= len(candidate_regions):
            break
    return [r["id"] for r in selected]


def format_structured_region_text(regions):
    lines = []
    for region in regions:
        cx, cy, bw, bh = region.get("global_box_norm", [0.0, 0.0, 0.0, 0.0])
        neighbors = region.get("neighbor_ids", [])
        neighbors_str = ",".join(neighbors) if neighbors else "-"
        lines.append(
            f"[Patch {region['id']} | role={region.get('role', 'context')} | grid=({region['row']},{region['col']}) | "
            f"box=({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}) | neighbors={{{neighbors_str}}} | scale={region.get('scale_token', 'grid')}]"
        )
    return "\n".join(lines)


class SemanticEncoder:
    def __init__(self, model_path, batch_size=12):
        self.model_path = model_path
        self.batch_size = max(1, int(batch_size))
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = next(self.model.parameters()).device
        self.text_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _extract_feature_tensor(feat):
        if isinstance(feat, torch.Tensor):
            return feat
        if isinstance(feat, (list, tuple)) and feat and isinstance(feat[0], torch.Tensor):
            return feat[0]
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            val = getattr(feat, attr, None)
            if isinstance(val, torch.Tensor):
                if attr == "last_hidden_state" and val.ndim >= 3:
                    return val[:, 0, :]
                return val
        if isinstance(feat, dict):
            for key in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
                val = feat.get(key)
                if isinstance(val, torch.Tensor):
                    if key == "last_hidden_state" and val.ndim >= 3:
                        return val[:, 0, :]
                    return val
        raise TypeError(f"Unsupported embedding output type: {type(feat)}")

    def _encode_text(self, text):
        if text in self.text_cache:
            return self.text_cache[text]
        with torch.no_grad():
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self.model.get_text_features(**inputs) if hasattr(self.model, "get_text_features") else self.model(**inputs)
            feat = self._extract_feature_tensor(feat)
            feat = self._normalize(feat.float()).squeeze(0).cpu()
        self.text_cache[text] = feat
        return feat

    def _encode_images(self, images):
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feat = self.model.get_image_features(**inputs) if hasattr(self.model, "get_image_features") else self.model(**inputs)
                feat = self._extract_feature_tensor(feat)
                feats.append(self._normalize(feat.float()).cpu())
        return torch.cat(feats, dim=0) if feats else torch.empty(0, 1)


def encode_patches_semantic(semantic_encoder, image, regions):
    crops = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        crops.append(image.crop((x1, y1, x2, y2)).convert("RGB"))
    feats = semantic_encoder._encode_images(crops)
    return {region["id"]: feats[i] for i, region in enumerate(regions)}


def encode_question_semantic(semantic_encoder, question):
    return semantic_encoder._encode_text(question)


def encode_global_semantic(semantic_encoder, plain_thumb):
    feats = semantic_encoder._encode_images([plain_thumb.convert("RGB")])
    return feats[0] if feats.numel() > 0 else None


def save_patch_bank_cache(cache_path, payload):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    serializable = dict(payload)
    serializable["patch_embeddings"] = {k: v.tolist() for k, v in payload["patch_embeddings"].items()}
    if payload.get("global_embedding") is not None:
        serializable["global_embedding"] = payload["global_embedding"].tolist()
    torch.save(serializable, cache_path)


def load_or_build_patch_bank(semantic_encoder, image_path, image, plain_thumb, regions, rows, cols, overlap_ratio, cache_dir):
    stat = os.stat(image_path)
    key = f"{image_path}|{stat.st_mtime_ns}|{stat.st_size}|{rows}|{cols}|{overlap_ratio:.4f}|{semantic_encoder.model_path}"
    cache_key = hashlib.md5(key.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
    if os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        patch_embeddings = {k: torch.tensor(v, dtype=torch.float32) for k, v in payload["patch_embeddings"].items()}
        global_embedding = payload.get("global_embedding")
        if global_embedding is not None:
            global_embedding = torch.tensor(global_embedding, dtype=torch.float32)
        return {"patch_embeddings": patch_embeddings, "global_embedding": global_embedding, "cache_hit": True, "cache_path": cache_path}

    patch_embeddings = encode_patches_semantic(semantic_encoder, image, regions)
    global_embedding = encode_global_semantic(semantic_encoder, plain_thumb)
    payload = {"patch_embeddings": patch_embeddings, "global_embedding": global_embedding}
    save_patch_bank_cache(cache_path, payload)
    return {"patch_embeddings": patch_embeddings, "global_embedding": global_embedding, "cache_hit": False, "cache_path": cache_path}


def retrieve_semantic_regions(regions, patch_embeddings, question_embedding, global_embedding, top_k, global_weight):
    region_ids = [region["id"] for region in regions]
    valid_pairs = [(rid, patch_embeddings[rid]) for rid in region_ids if rid in patch_embeddings]
    if not valid_pairs:
        return [], {}, {}
    ids, feats = zip(*valid_pairs)
    score_device = question_embedding.device if isinstance(question_embedding, torch.Tensor) else "cpu"
    patch_mat = torch.stack(feats, dim=0).to(score_device, non_blocking=True)
    q = question_embedding.to(score_device, non_blocking=True)
    scores = patch_mat @ q
    if global_embedding is not None:
        g = global_embedding.to(score_device, non_blocking=True)
        scores = scores + global_weight * (patch_mat @ g)
    scores_cpu = scores.detach().float().cpu()
    sorted_idx = torch.argsort(scores_cpu, descending=True)
    sorted_ids = [ids[i] for i in sorted_idx.tolist()]
    selected_ids = sorted_ids[:max(1, int(top_k))]
    semantic_scores = {ids[i]: float(scores_cpu[i]) for i in range(len(ids))}
    retrieval_rank = {rid: rank for rank, rid in enumerate(sorted_ids)}
    return selected_ids, semantic_scores, retrieval_rank


def call_qwen(processor, model, images, prompt, gen_kwargs):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [[{"role": "user", "content": content}]]
    text_inputs = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text_inputs, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)
    generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def extract_region_ids(raw_text, valid_ids):
    valid_set = set(valid_ids)
    found = []
    json_match = re.search(r"\{[\s\S]*?\}", raw_text)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            regions = obj.get("regions", [])
            if isinstance(regions, str):
                regions = [regions]
            for region in regions:
                region = str(region).upper().strip()
                if region in valid_set and region not in found:
                    found.append(region)
        except Exception:
            pass
    for match in re.findall(r"R\d+", raw_text.upper()):
        if match in valid_set and match not in found:
            found.append(match)
    return found


def select_ids_by_position_phrase(question, regions, rows, cols, limit):
    text = question.lower()

    def rows_for(top_words, mid_words, bottom_words):
        if any(w in text for w in top_words):
            return [0]
        if any(w in text for w in bottom_words):
            return [rows - 1]
        if any(w in text for w in mid_words):
            return [rows // 2]
        return list(range(rows))

    def cols_for(left_words, mid_words, right_words):
        if any(w in text for w in left_words):
            return [0]
        if any(w in text for w in right_words):
            return [cols - 1]
        if any(w in text for w in mid_words):
            return [cols // 2]
        return list(range(cols))

    row_targets = rows_for(["upper", "top", "north"], ["middle", "center", "centre"], ["lower", "bottom", "south"])
    col_targets = cols_for(["left", "west"], ["middle", "center", "centre"], ["right", "east"])
    if len(row_targets) == rows and len(col_targets) == cols:
        return []
    pairs = {(r, c) for r in row_targets for c in col_targets}
    out = [region["id"] for region in regions if (region["row"], region["col"]) in pairs]
    return out[:limit]


def expand_neighbors(selected_ids, regions, rows, cols, enabled):
    if not enabled:
        return selected_ids
    region_map = {region["id"]: region for region in regions}
    selected = []
    seen = set()
    for region_id in selected_ids:
        if region_id not in region_map:
            continue
        if region_id not in seen:
            selected.append(region_id)
            seen.add(region_id)
        row = region_map[region_id]["row"]
        col = region_map[region_id]["col"]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = row + dr
                nc = col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                cand = f"R{nr * cols + nc + 1}"
                if cand not in seen:
                    selected.append(cand)
                    seen.add(cand)
    return selected


def fallback_region_ids(question, regions, rows, cols, limit):
    ids = select_ids_by_position_phrase(question, regions, rows, cols, limit)
    if ids:
        return ids
    center_candidates = []
    target_row = rows // 2
    target_col = cols // 2
    for region in regions:
        dist = abs(region["row"] - target_row) + abs(region["col"] - target_col)
        center_candidates.append((dist, region["id"]))
    center_candidates.sort()
    return [region_id for _, region_id in center_candidates[:limit]]


def build_router_prompt(question, category, max_regions, region_metadata_text):
    return (
        "You are routing attention for remote sensing VQA.\n"
        "The image is a labeled global thumbnail divided into regions R1...Rn.\n"
        f"Question category: {category}\n"
        f"Question: {question}\n"
        f"Select at most {max_regions} region IDs that are most useful for answering the question.\n"
        "Return JSON only with this schema:\n"
        '{"regions": ["R1", "R2"], "reason": "short reason"}\n'
        "Available regions:\n"
        f"{region_metadata_text}"
    )


def build_evidence_board(selected_regions, patch_size):
    if not selected_regions:
        raise ValueError("selected_regions must not be empty")
    n = len(selected_regions)
    cols = min(3, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    padding = 8
    label_h = 24
    board_w = cols * patch_size + (cols + 1) * padding
    board_h = rows * (patch_size + label_h) + (rows + 1) * padding
    board = PILImage.new("RGB", (board_w, board_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    for idx, region in enumerate(selected_regions):
        row = idx // cols
        col = idx % cols
        x = padding + col * (patch_size + padding)
        y = padding + row * (patch_size + label_h + padding)
        patch = region["image"].resize((patch_size, patch_size), PILImage.BILINEAR)
        board.paste(patch, (x, y + label_h))
        label = f"{region['id']} | {sanitize_rel_pos(region['relative_position'])}"
        draw.rectangle((x, y, x + patch_size, y + label_h), fill=(25, 25, 25))
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255), font=font)
    return board


def build_topology_preserving_board(selected_regions, patch_size):
    if not selected_regions:
        raise ValueError("selected_regions must not be empty")
    unique_rows = sorted({region["row"] for region in selected_regions})
    unique_cols = sorted({region["col"] for region in selected_regions})
    row_map = {row: i for i, row in enumerate(unique_rows)}
    col_map = {col: i for i, col in enumerate(unique_cols)}
    rows = len(unique_rows)
    cols = len(unique_cols)
    padding = 8
    label_h = 24
    board_w = cols * patch_size + (cols + 1) * padding
    board_h = rows * (patch_size + label_h) + (rows + 1) * padding
    board = PILImage.new("RGB", (board_w, board_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    sorted_regions = sorted(selected_regions, key=lambda x: (x["row"], x["col"]))
    for region in sorted_regions:
        row = row_map[region["row"]]
        col = col_map[region["col"]]
        x = padding + col * (patch_size + padding)
        y = padding + row * (patch_size + label_h + padding)
        patch = region["image"].resize((patch_size, patch_size), PILImage.BILINEAR)
        board.paste(patch, (x, y + label_h))
        role_short = "A" if region.get("role") == "anchor" else "S"
        label = f"{region['id']}|{role_short}|r{region['row']}c{region['col']}"
        draw.rectangle((x, y, x + patch_size, y + label_h), fill=(25, 25, 25))
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255), font=font)
    board_layout = {
        "type": "topology_preserving_compact",
        "grid_rows": rows,
        "grid_cols": cols,
        "source_rows": unique_rows,
        "source_cols": unique_cols,
        "placements": [{"id": region["id"], "src_row": region["row"], "src_col": region["col"], "board_row": row_map[region["row"]], "board_col": col_map[region["col"]]} for region in sorted_regions],
    }
    return board, board_layout


def build_final_prompt(question, category, selected_regions, use_structured_metadata=True):
    if use_structured_metadata:
        region_text = format_structured_region_text(selected_regions)
    else:
        lines = []
        for region in selected_regions:
            x1, y1, x2, y2 = region["bbox"]
            lines.append(f"{region['id']}: position={sanitize_rel_pos(region['relative_position'])}, bbox=[{x1},{y1},{x2},{y2}]")
        region_text = "\n".join(lines)
    return (
        "Answer the question about the remote sensing image.\n"
        "You are given two images:\n"
        "1. A global thumbnail for overall scene layout.\n"
        "2. An evidence board with selected high-resolution regions.\n"
        f"Category: {category}\n"
        f"Question: {question}\n"
        "Region metadata:\n"
        f"{region_text}\n"
        "Use both global and local evidence jointly.\n"
        "Return only a short answer phrase in this format: <answer>...</answer>\n"
    )


def infer_one(processor, model, semantic_encoder, image_path, question, category, args, gen_kwargs):
    image = PILImage.open(image_path).convert("RGB")
    routing_info = {
        "routing_enabled": not args.disable_routing,
        "routing_mode": "thumbnail_evidence_v1" if not args.disable_routing else "baseline_single_image",
    }
    if args.disable_routing:
        image = resize_long_side(image, args.resize_size)
        prompt = (
            "Answer the question about the remote sensing image. "
            "Return only a short word or phrase.\n"
            f"Question: {question}\n"
            "Format: <answer>your short answer</answer>"
        )
        return call_qwen(processor, model, [image], prompt, gen_kwargs), routing_info

    plain_thumb = resize_long_side(image.copy(), args.resize_size)
    routed_thumb = resize_long_side(image.copy(), args.router_resize_size)
    routed_thumb = draw_labeled_grid(routed_thumb, args.routing_grid_rows, args.routing_grid_cols)
    regions = make_grid_regions(image.size[0], image.size[1], args.routing_grid_rows, args.routing_grid_cols, args.routing_overlap_ratio)
    max_regions = max(1, int(args.router_max_regions))
    router_raw = ""
    semantic_scores = {}
    retrieval_rank = {}
    cache_hit = False

    use_semantic_now = bool(args.use_semantic_support and semantic_encoder is not None)
    if use_semantic_now:
        try:
            cache_dir = args.semantic_cache_dir or os.path.join(args.save_path, "semantic_cache")
            patch_bank = load_or_build_patch_bank(
                semantic_encoder=semantic_encoder,
                image_path=image_path,
                image=image,
                plain_thumb=plain_thumb,
                regions=regions,
                rows=args.routing_grid_rows,
                cols=args.routing_grid_cols,
                overlap_ratio=args.routing_overlap_ratio,
                cache_dir=cache_dir,
            )
            question_emb = encode_question_semantic(semantic_encoder, question)
            selected_ids, semantic_scores, retrieval_rank = retrieve_semantic_regions(
                regions=regions,
                patch_embeddings=patch_bank["patch_embeddings"],
                question_embedding=question_emb,
                global_embedding=patch_bank.get("global_embedding"),
                top_k=max_regions,
                global_weight=args.semantic_global_weight,
            )
            routing_info["fallback_triggered"] = False
            routing_info["retrieval_mode"] = "semantic"
            cache_hit = patch_bank.get("cache_hit", False)
        except Exception as e:
            logger.warning("Semantic retrieval failed on sample, fallback to router mode: {}", str(e))
            routing_info["semantic_error"] = str(e)
            use_semantic_now = False

    if not use_semantic_now:
        region_metadata_text = build_region_metadata_text(regions)
        router_prompt = build_router_prompt(question=question, category=category, max_regions=max_regions, region_metadata_text=region_metadata_text)
        router_raw = call_qwen(processor, model, [routed_thumb], router_prompt, gen_kwargs)
        valid_ids = [region["id"] for region in regions]
        selected_ids = extract_region_ids(router_raw, valid_ids)
        if not selected_ids:
            selected_ids = fallback_region_ids(question, regions, args.routing_grid_rows, args.routing_grid_cols, max_regions)
            routing_info["fallback_triggered"] = True
        else:
            routing_info["fallback_triggered"] = False
        routing_info["retrieval_mode"] = "router"

    selected_ids = selected_ids[:max_regions]
    expanded_ids = expand_neighbors(selected_ids, regions, args.routing_grid_rows, args.routing_grid_cols, args.neighbor_expand)
    expanded_ids = expanded_ids[: max(max_regions + 2, args.support_set_budget + 4)]

    router_rank_map = {region_id: rank for rank, region_id in enumerate(selected_ids)}
    region_rank_fallback = len(router_rank_map) + 10
    enriched_all_regions = enrich_region_metadata(
        regions=regions,
        rows=args.routing_grid_rows,
        cols=args.routing_grid_cols,
        image_w=image.size[0],
        image_h=image.size[1],
        anchor_ids=selected_ids,
    )
    for region in enriched_all_regions:
        region["router_rank"] = router_rank_map.get(region["id"], region_rank_fallback)
        region["semantic_score"] = semantic_scores.get(region["id"], 0.0)
        region["retrieval_rank"] = retrieval_rank.get(region["id"], 10**9)

    candidate_id_set = set(expanded_ids)
    candidate_regions = [r for r in enriched_all_regions if r["id"] in candidate_id_set]
    if args.use_minimal_support_set:
        support_ids = select_minimal_support_set(
            candidate_regions=candidate_regions,
            support_set_budget=args.support_set_budget,
            support_iou_threshold=args.support_iou_threshold,
            semantic_score_weight=args.semantic_score_weight,
            anchor_bonus=args.semantic_anchor_bonus,
        )
    else:
        support_ids = [r["id"] for r in candidate_regions[: max(1, args.support_set_budget)]]

    if not support_ids:
        support_ids = selected_ids[:1] if selected_ids else [regions[len(regions) // 2]["id"]]

    enriched_all_regions = enrich_region_metadata(
        regions=enriched_all_regions,
        rows=args.routing_grid_rows,
        cols=args.routing_grid_cols,
        image_w=image.size[0],
        image_h=image.size[1],
        anchor_ids=selected_ids,
        support_ids=support_ids,
    )
    support_id_set = set(support_ids)
    support_regions = crop_regions(image, [r for r in enriched_all_regions if r["id"] in support_id_set])
    support_regions.sort(key=lambda x: (x["row"], x["col"]))
    if args.use_topology_board:
        evidence_board, board_layout = build_topology_preserving_board(support_regions, args.evidence_patch_size)
    else:
        evidence_board = build_evidence_board(support_regions, args.evidence_patch_size)
        board_layout = {"type": "sequential_grid", "grid_rows": None, "grid_cols": None, "placements": []}

    final_prompt = build_final_prompt(question, category, support_regions, use_structured_metadata=args.use_structured_metadata)
    raw_output = call_qwen(processor, model, [plain_thumb, evidence_board], final_prompt, gen_kwargs)

    routing_info.update(
        {
            "router_raw": router_raw,
            "selected_regions": selected_ids,
            "expanded_regions": expanded_ids,
            "candidate_regions": [r["id"] for r in candidate_regions],
            "support_regions": support_ids,
            "evidence_budget": args.support_set_budget,
            "board_layout": board_layout,
            "semantic_enabled": use_semantic_now,
            "semantic_topk": selected_ids,
            "semantic_scores": semantic_scores,
            "cache_hit": cache_hit,
            "retrieval_mode": routing_info.get("retrieval_mode", "router"),
            "n_selected_regions": len(selected_ids),
            "n_candidate_regions": len(candidate_regions),
            "n_support_regions": len(support_regions),
            "n_evidence_regions": len(support_regions),
            "grid_rows": args.routing_grid_rows,
            "grid_cols": args.routing_grid_cols,
            "neighbor_expand": args.neighbor_expand,
            "use_minimal_support_set": args.use_minimal_support_set,
            "use_structured_metadata": args.use_structured_metadata,
            "use_topology_board": args.use_topology_board,
        }
    )
    return raw_output, routing_info


def load_model_with_fallback(model_path, requested_attn):
    attn_candidates = [requested_attn, "sdpa", "eager"]
    seen = set()
    attn_candidates = [x for x in attn_candidates if not (x in seen or seen.add(x))]
    last_err = None
    for attn_impl in attn_candidates:
        try:
            logger.info("Trying to load model with attn_implementation={}", attn_impl)
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
            )
            model.eval()
            logger.info("Model loaded with attn_implementation={}", attn_impl)
            return model, attn_impl
        except Exception as e:
            last_err = e
            logger.warning("Load failed with attn_implementation={}: {}", attn_impl, str(e))
    raise RuntimeError(f"All attention implementations failed. Last error: {last_err}")


def source_from_qid(qid):
    qid = str(qid)
    if "_" not in qid:
        return "unknown"
    return qid.split("_", 1)[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="/WeaveEarth/ckpts/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--dataset_jsonl_path", type=str, default="/WeaveEarth/datars/LRS-VQA/LRS_VQA_merged.jsonl")
    parser.add_argument("--dataset_root", type=str, default="/WeaveEarth/datars/LRS-VQA")
    parser.add_argument("--save_path", type=str, default="/WeaveEarth/output/LRS-VQA-routed")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--resize_size", type=int, default=2048)
    parser.add_argument("--sample_limit", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--filter_category", type=str, default="")
    parser.add_argument("--filter_size_bin", type=str, default="")
    parser.add_argument("--strict_image_exists", action="store_true", default=False)

    parser.add_argument("--disable_routing", action="store_true", default=False)
    parser.add_argument("--routing_grid_rows", type=int, default=6)
    parser.add_argument("--routing_grid_cols", type=int, default=6)
    parser.add_argument("--routing_overlap_ratio", type=float, default=0.1)
    parser.add_argument("--router_max_regions", type=int, default=8)
    parser.add_argument("--neighbor_expand", action="store_true", default=True)
    parser.add_argument("--no_neighbor_expand", dest="neighbor_expand", action="store_false")
    parser.add_argument("--use_minimal_support_set", action="store_true", default=True)
    parser.add_argument("--no_minimal_support_set", dest="use_minimal_support_set", action="store_false")
    parser.add_argument("--use_structured_metadata", action="store_true", default=True)
    parser.add_argument("--no_structured_metadata", dest="use_structured_metadata", action="store_false")
    parser.add_argument("--use_topology_board", action="store_true", default=True)
    parser.add_argument("--no_topology_board", dest="use_topology_board", action="store_false")
    parser.add_argument("--support_set_budget", type=int, default=6)
    parser.add_argument("--support_iou_threshold", type=float, default=0.4)
    parser.add_argument("--evidence_patch_size", type=int, default=448)
    parser.add_argument("--router_resize_size", type=int, default=2048)

    parser.add_argument("--use_semantic_support", action="store_true", default=True)
    parser.add_argument("--no_semantic_support", dest="use_semantic_support", action="store_false")
    parser.add_argument("--semantic_model_path", type=str, default="/WeaveEarth/ckpts/siglip2-so400m-patch16-naflex")
    parser.add_argument("--semantic_batch_size", type=int, default=6)
    parser.add_argument("--semantic_cache_dir", type=str, default="/WeaveEarth/output/LRS-VQA-routed/cache")
    parser.add_argument("--semantic_score_weight", type=float, default=1.0)
    parser.add_argument("--semantic_anchor_bonus", type=float, default=0.08)
    parser.add_argument("--semantic_global_weight", type=float, default=0.35)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_start_time = time.time()
    os.makedirs(args.save_path, exist_ok=True)

    logger.info("Loading model: {}", args.reasoning_model_path)
    model, used_attn_impl = load_model_with_fallback(args.reasoning_model_path, args.attn_implementation)
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    semantic_encoder = None
    semantic_ready = False
    if args.use_semantic_support:
        try:
            semantic_encoder = SemanticEncoder(model_path=args.semantic_model_path, batch_size=args.semantic_batch_size)
            semantic_ready = True
            logger.info("Semantic encoder loaded: {}", args.semantic_model_path)
        except Exception as e:
            semantic_encoder = None
            logger.warning("Semantic encoder load failed, fallback to router mode: {}", str(e))

    data = load_jsonl(args.dataset_jsonl_path)
    assert isinstance(data, list), "dataset jsonl should be parsed as a list"

    image_index = build_image_index(args.dataset_root)
    logger.info("Indexed {} local images from {}", len(image_index), args.dataset_root)

    start_idx = max(0, args.start_idx)
    end_idx = len(data) if args.end_idx < 0 else min(len(data), args.end_idx)
    records = []
    missing_images = 0

    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample, "use_cache": True}
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature

    iterator = range(start_idx, end_idx)
    for i in tqdm(iterator):
        if args.sample_limit > 0 and len(records) >= args.sample_limit:
            break
        item = data[i]
        if not keep_item(item, args):
            continue

        image_rel = item.get("image", "")
        image_path = resolve_image_path(image_rel, args.dataset_root, image_index)
        if not image_path:
            missing_images += 1
            if args.strict_image_exists:
                continue
            records.append(
                {
                    "idx": i,
                    "question_id": item.get("question_id", ""),
                    "source": source_from_qid(item.get("question_id", "")),
                    "category": item.get("category", ""),
                    "size_bin": item.get("size_bin", ""),
                    "abs_size": item.get("abs_size", ""),
                    "question": item.get("text", ""),
                    "gt": item.get("ground_truth", ""),
                    "pred": "",
                    "match_method": "missing_image",
                    "is_correct": False,
                    "raw_output": "",
                    "image_rel": image_rel,
                    "image_path": "",
                    "routing": {"routing_enabled": not args.disable_routing, "missing_image": True},
                }
            )
            continue

        question = str(item.get("text", "")).strip()
        gt = str(item.get("ground_truth", "")).strip()
        category = str(item.get("category", "")).strip().lower()

        try:
            raw_output, routing_info = infer_one(
                processor=processor,
                model=model,
                semantic_encoder=semantic_encoder,
                image_path=image_path,
                question=question,
                category=category,
                args=args,
                gen_kwargs=gen_kwargs,
            )
        except Exception as e:
            if "wrap_triton" in str(e):
                raise RuntimeError(
                    "Inference hit torch/flash-attn incompatibility: "
                    "'torch.library.wrap_triton' not found. "
                    "Please rerun with --attn_implementation sdpa or eager."
                ) from e
            raw_output = f"[INFER_ERROR] {str(e)}"
            routing_info = {"routing_enabled": not args.disable_routing, "infer_error": str(e)}

        pred, extract_method = extract_model_answer(raw_output)
        is_correct, pred_norm, gt_norm, match_method = is_match(pred, gt)

        records.append(
            {
                "idx": i,
                "question_id": item.get("question_id", ""),
                "source": source_from_qid(item.get("question_id", "")),
                "category": item.get("category", ""),
                "size_bin": item.get("size_bin", ""),
                "abs_size": item.get("abs_size", ""),
                "question": question,
                "gt": gt,
                "pred": pred,
                "gt_norm": gt_norm,
                "pred_norm": pred_norm,
                "extract_method": extract_method,
                "match_method": match_method,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "image_rel": image_rel,
                "image_path": image_path,
                "routing": routing_info,
            }
        )

    valid_records = [r for r in records if r.get("match_method") != "missing_image"]
    total = len(valid_records)
    correct = sum(int(r["is_correct"]) for r in valid_records)
    overall_acc = correct / total if total > 0 else 0.0
    benchmark_total_time = time.time() - benchmark_start_time
    avg_time_per_sample = benchmark_total_time / total if total > 0 else 0.0

    category_stats = group_acc([{"group": r.get("category", ""), "is_correct": r["is_correct"]} for r in valid_records])
    size_bin_stats = group_acc([{"group": r.get("size_bin", ""), "is_correct": r["is_correct"]} for r in valid_records])
    abs_size_stats = group_acc([{"group": r.get("abs_size", ""), "is_correct": r["is_correct"]} for r in valid_records])
    source_stats = group_acc([{"group": r.get("source", ""), "is_correct": r["is_correct"]} for r in valid_records])

    summary = {
        "dataset_jsonl_path": args.dataset_jsonl_path,
        "dataset_root": args.dataset_root,
        "model_path": args.reasoning_model_path,
        "attn_implementation": used_attn_impl,
        "sample_window": {"start_idx": start_idx, "end_idx": end_idx},
        "sample_limit": args.sample_limit,
        "filter_category": args.filter_category,
        "filter_size_bin": args.filter_size_bin,
        "routing_enabled": not args.disable_routing,
        "routing_grid_rows": args.routing_grid_rows,
        "routing_grid_cols": args.routing_grid_cols,
        "routing_overlap_ratio": args.routing_overlap_ratio,
        "router_max_regions": args.router_max_regions,
        "neighbor_expand": args.neighbor_expand,
        "use_minimal_support_set": args.use_minimal_support_set,
        "use_structured_metadata": args.use_structured_metadata,
        "use_topology_board": args.use_topology_board,
        "support_set_budget": args.support_set_budget,
        "support_iou_threshold": args.support_iou_threshold,
        "use_semantic_support": args.use_semantic_support,
        "semantic_model_path": args.semantic_model_path,
        "semantic_batch_size": args.semantic_batch_size,
        "semantic_cache_dir": args.semantic_cache_dir,
        "semantic_score_weight": args.semantic_score_weight,
        "semantic_anchor_bonus": args.semantic_anchor_bonus,
        "semantic_global_weight": args.semantic_global_weight,
        "semantic_ready": semantic_ready,
        "evidence_patch_size": args.evidence_patch_size,
        "n_records_total": len(records),
        "n_records_evaluated": total,
        "n_correct": correct,
        "overall_acc": overall_acc,
        "benchmark_total_time_sec": benchmark_total_time,
        "avg_time_per_sample_sec": avg_time_per_sample,
        "n_missing_images": missing_images,
        "acc_by_source": source_stats,
        "acc_by_category": category_stats,
        "acc_by_size_bin": size_bin_stats,
        "acc_by_abs_size": abs_size_stats,
    }

    pred_path = os.path.join(args.save_path, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = os.path.join(args.save_path, "results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Saved predictions to {}", pred_path)
    logger.info("Saved summary to {}", summary_path)
    logger.info("Overall Acc: {:.4f} ({}/{})", overall_acc, correct, total)
    logger.info("Benchmark Total Time: {:.2f} sec", benchmark_total_time)
    logger.info("Avg Time per Sample: {:.4f} sec", avg_time_per_sample)


if __name__ == "__main__":
    main()

