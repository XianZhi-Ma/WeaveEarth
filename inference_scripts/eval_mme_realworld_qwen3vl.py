import argparse
import hashlib
import json
import logging
import os
import re
import time
import warnings
from collections import defaultdict
from typing import Dict

import difflib
import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

from qwen_vl_utils import process_vision_info

try:
    from loguru import logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    class _CompatLogger:
        def __init__(self):
            self._logger = logging.getLogger("eval_mme_realworld_qwen3vl")

        def info(self, msg, *args):
            self._logger.info(msg.format(*args))

        def warning(self, msg, *args):
            self._logger.warning(msg.format(*args))

        def error(self, msg, *args):
            self._logger.error(msg.format(*args))

    logger = _CompatLogger()

# Allow loading ultra-high-resolution remote sensing images.
PILImage.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)

MAIN_MAX_NEW_TOKENS = 1024
MAIN_DO_SAMPLE = False
MAIN_RESIZE_SIZE = 1024
MAIN_ROUTER_RESIZE_SIZE = 2048
MAIN_ROUTING_GRID_ROWS = 6
MAIN_ROUTING_GRID_COLS = 6
MAIN_ROUTING_OVERLAP_RATIO = 0.1
MAIN_ROUTER_MAX_REGIONS = 8
MAIN_SUPPORT_SET_BUDGET = 6
MAIN_SUPPORT_IOU_THRESHOLD = 0.4
MAIN_SEMANTIC_SCORE_WEIGHT = 1.0
MAIN_SEMANTIC_ANCHOR_BONUS = 0.08
MAIN_SEMANTIC_GLOBAL_WEIGHT = 0.35
MAIN_EVIDENCE_PATCH_SIZE = 448


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="/MegaEarth/ckpts/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/MegaEarth/datars/MME-RealWorld/MME_RealWorld.json",
    )
    parser.add_argument("--dataset_root", type=str, default="/MegaEarth/datars/MME-RealWorld")
    parser.add_argument("--save_path", type=str, default=" ")
    parser.add_argument("--sample_limit", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--filter_task", type=str, default="")
    parser.add_argument("--filter_subtask", type=str, default="")
    parser.add_argument("--filter_category", type=str, default="")
    parser.add_argument("--strict_image_exists", action="store_true", default=False)

    parser.add_argument(
        "--semantic_model_path",
        type=str,
        default="/MegaEarth/ckpts/siglip2-so400m-patch16-naflex",
    )
    parser.add_argument("--semantic_batch_size", type=int, default=6)
    parser.add_argument("--semantic_cache_dir", type=str, default="/MegaEarth/output/MME-RealWorld-siglip-retrieval/cache")
    parser.add_argument("--save_debug_images", action="store_true", default=False)
    return parser.parse_args()


def normalize_option_text(text):
    text = text.strip()
    text = re.sub(r"^\(?[A-Ea-e]\)?[\.、:\s-]*", "", text)
    return text.strip().lower()


def parse_gt_letter(gt):
    if gt is None:
        return ""
    gt = str(gt).strip().upper()
    match = re.search(r"[A-E]", gt)
    return match.group(0) if match else ""


def parse_model_choice(raw_text, answer_choices):
    text = raw_text.strip()
    text = (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
        .replace("Ａ", "A")
        .replace("Ｂ", "B")
        .replace("Ｃ", "C")
        .replace("Ｄ", "D")
        .replace("Ｅ", "E")
    )

    answer_tag = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if answer_tag:
        text = answer_tag.group(1).strip()

    for candidate in [text, raw_text]:
        json_match = re.search(r"\{[\s\S]*?\}", candidate)
        if not json_match:
            continue
        json_str = json_match.group(0)
        try:
            obj = json.loads(json_str)
            for k in ["response", "answer", "final_answer", "choice"]:
                if k in obj:
                    val = str(obj[k]).upper()
                    m = re.search(r"[A-E]", val)
                    if m:
                        return m.group(0), "json"
        except Exception:
            pass

    patterns = [
        r"答案是[:\s]*\(?([A-Ea-e])\)?",
        r"答案[:：\s]*\(?([A-Ea-e])\)?",
        r"正确答案是[:\s]*\(?([A-Ea-e])\)?",
        r"正确答案[:\s]*\(?([A-Ea-e])\)?",
        r"选项[:：\s]*\(?([A-Ea-e])\)?",
        r"option[:：\s]*\(?([A-Ea-e])\)?",
        r"choice[:：\s]*\(?([A-Ea-e])\)?",
        r"^\s*\(?([A-Ea-e])\)?\s*$",
        r"\(([A-Ea-e])\)",
        r"\b([A-Ea-e])\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), "regex"

    if answer_choices:
        best_letter = ""
        best_score = -1.0
        norm_text = normalize_option_text(text)
        for choice in answer_choices:
            letter_match = re.search(r"[A-Ea-e]", choice)
            if not letter_match:
                continue
            letter = letter_match.group(0).upper()
            choice_body = normalize_option_text(choice)
            score = difflib.SequenceMatcher(None, norm_text, choice_body).ratio()
            if choice_body in norm_text:
                score += 0.25
            if score > best_score:
                best_score = score
                best_letter = letter
        if best_letter and best_score >= 0.35:
            return best_letter, "fuzzy"

    return "", "fail"


def build_image_index(dataset_root):
    index = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, _, files in os.walk(dataset_root):
        for name in files:
            if not name.lower().endswith(valid_exts):
                continue
            index[name] = os.path.join(root, name)
    return index


def resolve_image_path(image_rel_path, dataset_root, image_index):
    direct_path = os.path.join(dataset_root, image_rel_path)
    if os.path.exists(direct_path):
        return direct_path
    base = os.path.basename(image_rel_path)
    if base in image_index:
        return image_index[base]
    return ""


def keep_item(item, args):
    task = str(item.get("Task", ""))
    subtask = str(item.get("Subtask", ""))
    category = str(item.get("Category", ""))

    if args.filter_task and task not in {x.strip() for x in args.filter_task.split(",") if x.strip()}:
        return False
    if args.filter_subtask and subtask not in {
        x.strip() for x in args.filter_subtask.split(",") if x.strip()
    }:
        return False
    if args.filter_category and category not in {
        x.strip() for x in args.filter_category.split(",") if x.strip()
    }:
        return False
    return True


def group_acc(records):
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        grouped[r["group"]]["total"] += 1
        if r["is_correct"]:
            grouped[r["group"]]["correct"] += 1
    out = {}
    for k, v in grouped.items():
        total = v["total"]
        acc = v["correct"] / total if total > 0 else 0.0
        out[k] = {"correct": v["correct"], "total": total, "acc": acc}
    return out


def resize_long_side(image, target_long_side):
    if target_long_side <= 0:
        return image
    w, h = image.size
    if max(w, h) <= target_long_side:
        return image
    scale = target_long_side / max(w, h)
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), PILImage.BILINEAR)


def relative_position_from_indices(row, col, rows, cols):
    v = "top" if row < rows / 3 else "bottom" if row >= rows * 2 / 3 else "middle"
    h = "left" if col < cols / 3 else "right" if col >= cols * 2 / 3 else "center"
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return f"{h}-center"
    if h == "center":
        return f"{v}-center"
    return f"{v}-{h}"


def make_grid_regions(width, height, rows, cols, overlap_ratio):
    cell_w = width / cols
    cell_h = height / rows
    overlap_x = int(cell_w * overlap_ratio)
    overlap_y = int(cell_h * overlap_ratio)

    regions = []
    idx = 1
    for row in range(rows):
        for col in range(cols):
            x1 = max(0, int(round(col * cell_w)) - overlap_x)
            y1 = max(0, int(round(row * cell_h)) - overlap_y)
            x2 = min(width, int(round((col + 1) * cell_w)) + overlap_x)
            y2 = min(height, int(round((row + 1) * cell_h)) + overlap_y)
            regions.append(
                {
                    "id": f"R{idx}",
                    "idx": idx,
                    "row": row,
                    "col": col,
                    "bbox": [x1, y1, x2, y2],
                    "relative_position": relative_position_from_indices(row, col, rows, cols),
                }
            )
            idx += 1
    return regions


def crop_regions(image, regions):
    out = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        item = dict(region)
        item["image"] = image.crop((x1, y1, x2, y2))
        out.append(item)
    return out


def draw_labeled_grid(image, rows, cols):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, h = image.size
    cell_w = w / cols
    cell_h = h / rows
    idx = 1
    for row in range(rows):
        for col in range(cols):
            x1 = int(round(col * cell_w))
            y1 = int(round(row * cell_h))
            x2 = int(round((col + 1) * cell_w))
            y2 = int(round((row + 1) * cell_h))
            draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
            label = f"R{idx}"
            text_box = draw.textbbox((0, 0), label, font=font)
            tw = text_box[2] - text_box[0]
            th = text_box[3] - text_box[1]
            draw.rectangle((x1, y1, x1 + tw + 6, y1 + th + 4), fill=(255, 0, 0))
            draw.text((x1 + 3, y1 + 2), label, fill=(255, 255, 255), font=font)
            idx += 1
    return image


def bbox_iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = max(0, x12 - x11) * max(0, y12 - y11)
    area2 = max(0, x22 - x21) * max(0, y22 - y21)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def enrich_region_metadata(regions, rows, cols, image_w, image_h, anchor_ids=None, support_ids=None):
    anchor_ids = set(anchor_ids or [])
    support_ids = set(support_ids or [])
    region_map = {region["id"]: region for region in regions}
    scale_token = f"grid_{rows}x{cols}"
    enriched = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        cx = ((x1 + x2) / 2.0) / max(1, image_w)
        cy = ((y1 + y2) / 2.0) / max(1, image_h)
        bw = (x2 - x1) / max(1, image_w)
        bh = (y2 - y1) / max(1, image_h)
        neighbor_ids = []
        row = region["row"]
        col = region["col"]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                nc = col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                neighbor_id = f"R{nr * cols + nc + 1}"
                if neighbor_id in region_map:
                    neighbor_ids.append(neighbor_id)

        if region["id"] in anchor_ids:
            role = "anchor"
        elif region["id"] in support_ids:
            role = "support"
        else:
            role = "context"

        item = dict(region)
        item["global_box_norm"] = [cx, cy, bw, bh]
        item["neighbor_ids"] = neighbor_ids
        item["scale_token"] = scale_token
        item["role"] = role
        item["is_anchor"] = region["id"] in anchor_ids
        item["neighbor_distance"] = 0 if item["is_anchor"] else 1
        enriched.append(item)
    return enriched


def select_minimal_support_set(
    candidate_regions,
    support_set_budget,
    support_iou_threshold,
    semantic_score_weight=1.0,
    anchor_bonus=0.08,
):
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

            max_iou = 0.0
            if selected:
                max_iou = max(bbox_iou(region["bbox"], s["bbox"]) for s in selected)
            new_row_gain = 1 if region["row"] not in selected_rows else 0
            new_col_gain = 1 if region["col"] not in selected_cols else 0
            overlap_penalty = 1 if max_iou >= support_iou_threshold else 0
            semantic_score = float(region.get("semantic_score", 0.0))
            anchor_term = anchor_bonus if region.get("is_anchor", False) else 0.0
            # Semantic-aware selection with coverage/redundancy regularization.
            score = (
                semantic_score_weight * semantic_score
                + anchor_term
                + 2.5 * (new_row_gain + new_col_gain)
                - 2.0 * overlap_penalty
                - max_iou
            )
            key = (
                score,
                semantic_score,
                -max_iou,
                -region["neighbor_distance"],
                -region["row"],
                -region["col"],
            )
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
            f"[Patch {region['id']} | role={region.get('role', 'context')} | "
            f"grid=({region['row']},{region['col']}) | "
            f"box=({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}) | "
            f"neighbors={{{neighbors_str}}} | "
            f"scale={region.get('scale_token', 'grid')}]"
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
            if hasattr(self.model, "get_text_features"):
                feat = self.model.get_text_features(**inputs)
            else:
                feat = self.model(**inputs)
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
                if hasattr(self.model, "get_image_features"):
                    feat = self.model.get_image_features(**inputs)
                else:
                    feat = self.model(**inputs)
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


def encode_question_semantic(semantic_encoder, question, answer_choices):
    text = f"{question}\nOptions:\n" + "\n".join(answer_choices)
    return semantic_encoder._encode_text(text)


def encode_global_semantic(semantic_encoder, plain_thumb):
    feats = semantic_encoder._encode_images([plain_thumb.convert("RGB")])
    return feats[0] if feats.numel() > 0 else None


def save_patch_bank_cache(cache_path, payload):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    serializable = dict(payload)
    serializable["patch_embeddings"] = {
        k: v.tolist() for k, v in payload["patch_embeddings"].items()
    }
    if payload.get("global_embedding") is not None:
        serializable["global_embedding"] = payload["global_embedding"].tolist()
    torch.save(serializable, cache_path)


def load_or_build_patch_bank(
    semantic_encoder,
    image_path,
    image,
    plain_thumb,
    regions,
    rows,
    cols,
    overlap_ratio,
    cache_dir,
):
    stat = os.stat(image_path)
    key_str = (
        f"{image_path}|{stat.st_mtime_ns}|{stat.st_size}|"
        f"{rows}|{cols}|{overlap_ratio:.4f}|{semantic_encoder.model_path}"
    )
    cache_key = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        patch_embeddings = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in payload["patch_embeddings"].items()
        }
        global_embedding = payload.get("global_embedding")
        if global_embedding is not None:
            global_embedding = torch.tensor(global_embedding, dtype=torch.float32)
        return {
            "patch_embeddings": patch_embeddings,
            "global_embedding": global_embedding,
            "cache_hit": True,
            "cache_path": cache_path,
        }

    patch_embeddings = encode_patches_semantic(semantic_encoder, image, regions)
    global_embedding = encode_global_semantic(semantic_encoder, plain_thumb)
    payload = {
        "patch_embeddings": patch_embeddings,
        "global_embedding": global_embedding,
    }
    save_patch_bank_cache(cache_path, payload)
    return {
        "patch_embeddings": patch_embeddings,
        "global_embedding": global_embedding,
        "cache_hit": False,
        "cache_path": cache_path,
    }


def retrieve_semantic_regions(
    regions,
    patch_embeddings,
    question_embedding,
    global_embedding,
    top_k,
    global_weight,
):
    scored = []
    for region in regions:
        region_id = region["id"]
        patch_emb = patch_embeddings.get(region_id)
        if patch_emb is None:
            score_q = 0.0
            score_g = 0.0
        else:
            score_q = float(torch.dot(patch_emb, question_embedding))
            score_g = float(torch.dot(patch_emb, global_embedding)) if global_embedding is not None else 0.0
        total_score = score_q + global_weight * score_g
        scored.append(
            {
                "id": region_id,
                "semantic_score": total_score,
                "sim_question": score_q,
                "sim_global": score_g,
            }
        )

    scored.sort(key=lambda x: x["semantic_score"], reverse=True)
    selected_ids = [x["id"] for x in scored[:max(1, int(top_k))]]
    semantic_scores = {x["id"]: x["semantic_score"] for x in scored}
    retrieval_rank = {x["id"]: rank for rank, x in enumerate(scored)}
    return selected_ids, semantic_scores, retrieval_rank


def call_qwen(processor, model, images, prompt, gen_kwargs):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [[{"role": "user", "content": content}]]

    text_inputs = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text_inputs,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def expand_neighbors(selected_ids, regions, rows, cols):
    region_map = {region["id"]: region for region in regions}
    selected = []
    seen = set()
    for region_id in selected_ids:
        if region_id not in region_map:
            continue
        for cand in [region_id]:
            if cand not in seen:
                selected.append(cand)
                seen.add(cand)
        row = region_map[region_id]["row"]
        col = region_map[region_id]["col"]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = row + dr
                nc = col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                idx = nr * cols + nc + 1
                cand = f"R{idx}"
                if cand not in seen:
                    selected.append(cand)
                    seen.add(cand)
    return selected


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
        "placements": [
            {
                "id": region["id"],
                "src_row": region["row"],
                "src_col": region["col"],
                "board_row": row_map[region["row"]],
                "board_col": col_map[region["col"]],
            }
            for region in sorted_regions
        ],
    }
    return board, board_layout


def build_final_prompt(question, answer_choices, category, selected_regions):
    option_text = "\n".join(answer_choices)
    region_text = format_structured_region_text(selected_regions)
    return (
        "请根据遥感图像回答选择题。\n"
        "你会看到两张图：\n"
        "1. 全局缩略图，用于理解整体布局。\n"
        "2. evidence board，其中包含若干高分辨率证据区域。\n"
        f"问题类型：{category}\n"
        f"问题：{question}\n"
        f"选项：\n{option_text}\n"
        "证据区域元信息：\n"
        f"{region_text}\n"
        "要求：结合全局缩略图与局部证据共同推理，不要只依赖其中一张图。\n"
        "优先在 A/B/C/D 中选择，不要选择模糊选项 E。\n"
        "请先在<think></think>里简要推理，再在<answer></answer>里只输出一个选项字母（A/B/C/D/E）。"
    )


def maybe_save_debug_images(save_dir, stem, plain_thumb, routed_thumb, evidence_board):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    plain_thumb.save(os.path.join(save_dir, f"{stem}_thumb.png"))
    routed_thumb.save(os.path.join(save_dir, f"{stem}_routed_thumb.png"))
    evidence_board.save(os.path.join(save_dir, f"{stem}_evidence.png"))


def infer_one(
    processor,
    model,
    semantic_encoder,
    image_path,
    question,
    answer_choices,
    category,
    args,
    gen_kwargs,
    debug_stem="",
):
    image = PILImage.open(image_path).convert("RGB")
    routing_info = {
        "routing_enabled": True,
        "routing_mode": "thumbnail_evidence_v1",
    }

    plain_thumb = resize_long_side(image.copy(), MAIN_RESIZE_SIZE)
    routed_thumb = resize_long_side(image.copy(), MAIN_ROUTER_RESIZE_SIZE)
    routed_thumb = draw_labeled_grid(
        routed_thumb, MAIN_ROUTING_GRID_ROWS, MAIN_ROUTING_GRID_COLS
    )

    regions = make_grid_regions(
        image.size[0],
        image.size[1],
        MAIN_ROUTING_GRID_ROWS,
        MAIN_ROUTING_GRID_COLS,
        MAIN_ROUTING_OVERLAP_RATIO,
    )
    max_regions = MAIN_ROUTER_MAX_REGIONS
    semantic_scores = {}
    retrieval_rank = {}
    cache_hit = False

    if semantic_encoder is None:
        raise RuntimeError("Semantic encoder is required for the released WeaveEarth main method.")

    try:
        cache_dir = args.semantic_cache_dir or os.path.join(args.save_path, "semantic_cache")
        patch_bank = load_or_build_patch_bank(
            semantic_encoder=semantic_encoder,
            image_path=image_path,
            image=image,
            plain_thumb=plain_thumb,
            regions=regions,
            rows=MAIN_ROUTING_GRID_ROWS,
            cols=MAIN_ROUTING_GRID_COLS,
            overlap_ratio=MAIN_ROUTING_OVERLAP_RATIO,
            cache_dir=cache_dir,
        )
        question_emb = encode_question_semantic(semantic_encoder, question, answer_choices)
        selected_ids, semantic_scores, retrieval_rank = retrieve_semantic_regions(
            regions=regions,
            patch_embeddings=patch_bank["patch_embeddings"],
            question_embedding=question_emb,
            global_embedding=patch_bank.get("global_embedding"),
            top_k=max_regions,
            global_weight=MAIN_SEMANTIC_GLOBAL_WEIGHT,
        )
        routing_info["fallback_triggered"] = False
        routing_info["retrieval_mode"] = "semantic"
        cache_hit = patch_bank.get("cache_hit", False)
    except Exception as e:
        raise RuntimeError("Semantic retrieval failed; main-method inference cannot continue.") from e

    selected_ids = selected_ids[:max_regions]
    expanded_ids = expand_neighbors(
        selected_ids,
        regions,
        MAIN_ROUTING_GRID_ROWS,
        MAIN_ROUTING_GRID_COLS,
    )
    expanded_ids = expanded_ids[: max(max_regions + 2, MAIN_SUPPORT_SET_BUDGET + 4)]

    router_rank_map = {region_id: rank for rank, region_id in enumerate(selected_ids)}
    region_rank_fallback = len(router_rank_map) + 10
    enriched_all_regions = enrich_region_metadata(
        regions=regions,
        rows=MAIN_ROUTING_GRID_ROWS,
        cols=MAIN_ROUTING_GRID_COLS,
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

    support_ids = select_minimal_support_set(
        candidate_regions=candidate_regions,
        support_set_budget=MAIN_SUPPORT_SET_BUDGET,
        support_iou_threshold=MAIN_SUPPORT_IOU_THRESHOLD,
        semantic_score_weight=MAIN_SEMANTIC_SCORE_WEIGHT,
        anchor_bonus=MAIN_SEMANTIC_ANCHOR_BONUS,
    )

    if not support_ids:
        support_ids = selected_ids[:1] if selected_ids else [regions[len(regions) // 2]["id"]]

    enriched_all_regions = enrich_region_metadata(
        regions=enriched_all_regions,
        rows=MAIN_ROUTING_GRID_ROWS,
        cols=MAIN_ROUTING_GRID_COLS,
        image_w=image.size[0],
        image_h=image.size[1],
        anchor_ids=selected_ids,
        support_ids=support_ids,
    )
    support_id_set = set(support_ids)
    support_regions = crop_regions(image, [r for r in enriched_all_regions if r["id"] in support_id_set])
    support_regions.sort(key=lambda x: (x["row"], x["col"]))

    evidence_board, board_layout = build_topology_preserving_board(
        support_regions, MAIN_EVIDENCE_PATCH_SIZE
    )

    if args.save_debug_images and debug_stem:
        debug_dir = os.path.join(args.save_path, "debug_images")
        maybe_save_debug_images(debug_dir, debug_stem, plain_thumb, routed_thumb, evidence_board)

    final_prompt = build_final_prompt(
        question,
        answer_choices,
        category,
        support_regions,
    )
    raw_output = call_qwen(processor, model, [plain_thumb, evidence_board], final_prompt, gen_kwargs)

    routing_info.update(
        {
            "router_raw": "",
            "selected_regions": selected_ids,
            "expanded_regions": expanded_ids,
            "candidate_regions": [r["id"] for r in candidate_regions],
            "support_regions": support_ids,
            "evidence_budget": MAIN_SUPPORT_SET_BUDGET,
            "board_layout": board_layout,
            "semantic_enabled": True,
            "semantic_topk": selected_ids,
            "semantic_scores": semantic_scores,
            "cache_hit": cache_hit,
            "retrieval_mode": routing_info.get("retrieval_mode", "semantic"),
            "n_selected_regions": len(selected_ids),
            "n_candidate_regions": len(candidate_regions),
            "n_support_regions": len(support_regions),
            "n_evidence_regions": len(support_regions),
            "grid_rows": MAIN_ROUTING_GRID_ROWS,
            "grid_cols": MAIN_ROUTING_GRID_COLS,
            "neighbor_expand": True,
            "support_selection": "minimal_support_set",
            "metadata_format": "structured",
            "evidence_board": "topology_preserving",
            "region_metadata": [
                {
                    "id": region["id"],
                    "row": region["row"],
                    "col": region["col"],
                    "relative_position": region["relative_position"],
                    "bbox": region["bbox"],
                    "global_box_norm": region.get("global_box_norm"),
                    "neighbor_ids": region.get("neighbor_ids"),
                    "scale_token": region.get("scale_token"),
                    "role": region.get("role"),
                    "semantic_score": region.get("semantic_score", 0.0),
                    "retrieval_rank": region.get("retrieval_rank", 10**9),
                }
                for region in support_regions
            ],
            "support_region_metadata": [
                {
                    "id": region["id"],
                    "row": region["row"],
                    "col": region["col"],
                    "bbox": region["bbox"],
                    "global_box_norm": region.get("global_box_norm"),
                    "neighbor_ids": region.get("neighbor_ids"),
                    "scale_token": region.get("scale_token"),
                    "role": region.get("role"),
                    "semantic_score": region.get("semantic_score", 0.0),
                    "retrieval_rank": region.get("retrieval_rank", 10**9),
                }
                for region in support_regions
            ],
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


def main():
    args = parse_args()
    benchmark_start_time = time.time()
    os.makedirs(args.save_path, exist_ok=True)

    logger.info("Loading model: {}", args.reasoning_model_path)
    model, used_attn_impl = load_model_with_fallback(
        args.reasoning_model_path, args.attn_implementation
    )
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    try:
        semantic_encoder = SemanticEncoder(
            model_path=args.semantic_model_path,
            batch_size=args.semantic_batch_size,
        )
        semantic_ready = True
        logger.info("Semantic encoder loaded: {}", args.semantic_model_path)
    except Exception as e:
        raise RuntimeError(
            "Semantic encoder failed to load. The released script only exposes the "
            "WeaveEarth main method, which requires semantic support."
        ) from e

    with open(args.dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "dataset json should be a list"

    image_index = build_image_index(args.dataset_root)
    logger.info("Indexed {} local images from {}", len(image_index), args.dataset_root)

    start_idx = max(0, args.start_idx)
    end_idx = len(data) if args.end_idx < 0 else min(len(data), args.end_idx)
    records = []
    missing_images = 0
    parse_fail = 0

    gen_kwargs = {
        "max_new_tokens": MAIN_MAX_NEW_TOKENS,
        "do_sample": MAIN_DO_SAMPLE,
        "use_cache": True,
    }

    iterator = range(start_idx, end_idx)
    for i in tqdm(iterator):
        if args.sample_limit > 0 and len(records) >= args.sample_limit:
            break

        item = data[i]
        if not keep_item(item, args):
            continue

        image_rel = item.get("Image", "")
        image_path = resolve_image_path(image_rel, args.dataset_root, image_index)
        if not image_path:
            missing_images += 1
            if args.strict_image_exists:
                continue
            records.append(
                {
                    "idx": i,
                    "question_id": item.get("Question_id", ""),
                    "group_task": item.get("Task", ""),
                    "group_subtask": item.get("Subtask", ""),
                    "group_category": item.get("Category", ""),
                    "question": item.get("Text", ""),
                    "answer_choices": item.get("Answer choices", []),
                    "gt": parse_gt_letter(item.get("Ground truth", "")),
                    "pred": "",
                    "parse_method": "missing_image",
                    "is_correct": False,
                    "raw_output": "",
                    "image_rel": image_rel,
                    "image_path": "",
                }
            )
            continue

        question = str(item.get("Text", "")).strip()
        answer_choices = item.get("Answer choices", [])
        gt = parse_gt_letter(item.get("Ground truth", ""))
        category = str(item.get("Category", "")).strip().lower()
        debug_stem = f"{i:04d}_{os.path.splitext(os.path.basename(image_path))[0]}"

        try:
            raw_output, routing_info = infer_one(
                processor=processor,
                model=model,
                semantic_encoder=semantic_encoder,
                image_path=image_path,
                question=question,
                answer_choices=answer_choices,
                category=category,
                args=args,
                gen_kwargs=gen_kwargs,
                debug_stem=debug_stem,
            )
        except Exception as e:
            if "wrap_triton" in str(e):
                raise RuntimeError(
                    "Inference hit torch/flash-attn incompatibility: "
                    "'torch.library.wrap_triton' not found. "
                    "Please rerun with --attn_implementation sdpa or eager."
                ) from e
            raw_output = f"[INFER_ERROR] {str(e)}"
            routing_info = {"routing_enabled": True, "infer_error": str(e)}

        pred, parse_method = parse_model_choice(raw_output, answer_choices)
        if not pred:
            parse_fail += 1
        is_correct = bool(pred) and (pred == gt)

        records.append(
            {
                "idx": i,
                "question_id": item.get("Question_id", ""),
                "group_task": item.get("Task", ""),
                "group_subtask": item.get("Subtask", ""),
                "group_category": item.get("Category", ""),
                "question_type": item.get("Question Type", item.get("Question type", "")),
                "question": question,
                "answer_choices": answer_choices,
                "gt": gt,
                "pred": pred,
                "parse_method": parse_method,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "image_rel": image_rel,
                "image_path": image_path,
                "routing": routing_info,
            }
        )

    valid_records = [r for r in records if r.get("parse_method") != "missing_image"]
    total = len(valid_records)
    correct = sum(int(r["is_correct"]) for r in valid_records)
    overall_acc = correct / total if total > 0 else 0.0
    benchmark_total_time = time.time() - benchmark_start_time
    avg_time_per_sample = benchmark_total_time / total if total > 0 else 0.0

    task_stats = group_acc(
        [{"group": r["group_task"], "is_correct": r["is_correct"]} for r in valid_records]
    )
    subtask_stats = group_acc(
        [{"group": r["group_subtask"], "is_correct": r["is_correct"]} for r in valid_records]
    )
    category_stats = group_acc(
        [{"group": r["group_category"], "is_correct": r["is_correct"]} for r in valid_records]
    )

    summary = {
        "dataset_json_path": args.dataset_json_path,
        "dataset_root": args.dataset_root,
        "model_path": args.reasoning_model_path,
        "attn_implementation": used_attn_impl,
        "sample_window": {"start_idx": start_idx, "end_idx": end_idx},
        "sample_limit": args.sample_limit,
        "filter_task": args.filter_task,
        "filter_subtask": args.filter_subtask,
        "filter_category": args.filter_category,
        "max_new_tokens": MAIN_MAX_NEW_TOKENS,
        "do_sample": MAIN_DO_SAMPLE,
        "resize_size": MAIN_RESIZE_SIZE,
        "router_resize_size": MAIN_ROUTER_RESIZE_SIZE,
        "routing_enabled": True,
        "routing_grid_rows": MAIN_ROUTING_GRID_ROWS,
        "routing_grid_cols": MAIN_ROUTING_GRID_COLS,
        "routing_overlap_ratio": MAIN_ROUTING_OVERLAP_RATIO,
        "router_max_regions": MAIN_ROUTER_MAX_REGIONS,
        "neighbor_expand": True,
        "support_selection": "minimal_support_set",
        "metadata_format": "structured",
        "evidence_board": "topology_preserving",
        "support_set_budget": MAIN_SUPPORT_SET_BUDGET,
        "support_iou_threshold": MAIN_SUPPORT_IOU_THRESHOLD,
        "semantic_support": True,
        "semantic_model_path": args.semantic_model_path,
        "semantic_batch_size": args.semantic_batch_size,
        "semantic_cache_dir": args.semantic_cache_dir,
        "semantic_score_weight": MAIN_SEMANTIC_SCORE_WEIGHT,
        "semantic_anchor_bonus": MAIN_SEMANTIC_ANCHOR_BONUS,
        "semantic_global_weight": MAIN_SEMANTIC_GLOBAL_WEIGHT,
        "semantic_ready": semantic_ready,
        "evidence_patch_size": MAIN_EVIDENCE_PATCH_SIZE,
        "n_records_total": len(records),
        "n_records_evaluated": total,
        "n_correct": correct,
        "overall_acc": overall_acc,
        "benchmark_total_time_sec": benchmark_total_time,
        "avg_time_per_sample_sec": avg_time_per_sample,
        "n_missing_images": missing_images,
        "n_parse_fail": parse_fail,
        "acc_by_task": task_stats,
        "acc_by_subtask": subtask_stats,
        "acc_by_category": category_stats,
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
