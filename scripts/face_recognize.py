#!/usr/bin/env python3
"""
本地人脸识别脚本（统一特征矩阵版）
- YuNet 检测人脸 + 5 个关键点
- SFace 提取 512 维特征向量
- 统一特征矩阵 (features.npy) + 标签索引，搜索向量化
- 每人最多 10 个特征，超限删除与其他 9 个平均距离最远的
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SKILL_DIR = Path(__file__).resolve().parent.parent
FACE_DB_DIR = SKILL_DIR / "faces"
YUNET_MODEL = SKILL_DIR / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = SKILL_DIR / "models" / "face_recognition_sface_2021dec.onnx"

FEATURES_NPY = SKILL_DIR / "features.npy"
LABELS_JSON = SKILL_DIR / "labels.json"
META_JSON = SKILL_DIR / "meta.json"

SIMILARITY_THRESHOLD = 0.380 # 0.363
MAX_FEATURES_PER_PERSON = 10

# 全局检测器（懒加载）
_yunet_detector = None
_sface_recognizer = None


def _get_yunet(w, h):
    global _yunet_detector
    if _yunet_detector is None:
        _yunet_detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (w, h), 0.9, 0.3, 5000)
    else:
        _yunet_detector.setInputSize((w, h))
    return _yunet_detector


def _get_sface():
    global _sface_recognizer
    if _sface_recognizer is None:
        _sface_recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    return _sface_recognizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_faces(image_path: str):
    """YuNet 检测，返回 (img, [face_array, ...])"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, []
    h, w = img.shape[:2]
    detector = _get_yunet(w, h)
    _, faces = detector.detect(img)
    if faces is None:
        return img, []
    return img, faces


def extract_feature(img, face_array):
    """用 SFace 提取 512 维特征向量
    face_array: YuNet 原始输出，shape (15,) 或 (1,15)
    """
    recognizer = _get_sface()
    face_array = np.array(face_array).reshape(1, 15)
    aligned = recognizer.alignCrop(img, face_array)
    feat = recognizer.feature(aligned)
    return feat.flatten()


def get_face_roi(img, face_array, pad=0.2):
    """裁出人脸区域，用于保存排查小图"""
    x, y, w, h = face_array[0], face_array[1], face_array[2], face_array[3]
    px = int(w * pad)
    py = int(h * pad)
    h_img, w_img = img.shape[:2]
    x1 = max(0, int(x - px))
    y1 = max(0, int(y - py))
    x2 = min(w_img, int(x + w + px))
    y2 = min(h_img, int(y + h + py))
    roi = img[y1:y2, x1:x2]
    return cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)


def pairwise_distances(features):
    """计算特征向量的成对欧氏距离，用于多样性判断"""
    if features.shape[0] <= 1:
        return np.zeros((features.shape[0], features.shape[0]))
    f = features.astype(np.float32)
    sq = np.sum(f**2, axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(sq + sq.T - 2 * np.dot(f, f.T), 0))
    return dists


# ---------------------------------------------------------------------------
# Feature DB（统一矩阵版）
# ---------------------------------------------------------------------------
class FeatureDB:
    def __init__(self):
        self.features = None   # (N, 512) float32 或 None
        self.labels = []       # [{"name": "<person_name>", "id": 0}, ...]
        self.meta = {}         # {"<person_name>": {"ids": [0, 3]}, "next_id": 4}
        self._load()

    def _load(self):
        """优先加载统一矩阵，不存在则从旧格式迁移"""
        if FEATURES_NPY.exists() and LABELS_JSON.exists() and META_JSON.exists():
            self.features = np.load(FEATURES_NPY)
            with open(LABELS_JSON, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            with open(META_JSON, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            return
        self._migrate_from_legacy()

    def _migrate_from_legacy(self):
        """从 faces/{name}/face_*.npy 旧格式迁移到统一矩阵，并归一化"""
        features = []
        labels = []
        meta = {"next_id": 0}

        if FACE_DB_DIR.exists():
            for person_dir in sorted(FACE_DB_DIR.iterdir()):
                if not person_dir.is_dir():
                    continue
                name = person_dir.name
                npy_files = sorted(person_dir.glob("face_*.npy"))
                if not npy_files:
                    continue  # 过滤空目录
                meta[name] = {"ids": []}
                for npy_file in npy_files:
                    try:
                        feat = np.load(npy_file)
                        features.append(feat)
                        fid = meta["next_id"]
                        meta["next_id"] += 1
                        labels.append({"name": name, "id": fid})
                        meta[name]["ids"].append(fid)
                    except Exception:
                        pass

        if features:
            mat = np.stack(features, axis=0).astype(np.float32)
            self.features = self._l2_normalize(mat)
        else:
            self.features = np.zeros((0, 128), dtype=np.float32)
        self.labels = labels
        self.meta = meta
        self._save()

    def _save(self):
        """持久化统一矩阵 + 索引"""
        np.save(FEATURES_NPY, self.features)
        with open(LABELS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
        with open(META_JSON, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _l2_normalize(mat):
        """L2 归一化，避免除零"""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def search(self, query_feat):
        """向量化搜索，返回 (best_name, best_score)"""
        if self.features is None or self.features.shape[0] == 0:
            return "unknown", 0.0

        q = query_feat.astype(np.float32).flatten()
        q = q / np.linalg.norm(q)  # 归一化 query
        # 矩阵在 add/migrate 时已归一化，直接点乘 = cosine similarity
        scores = self.features @ q  # (N,)
        best_idx = int(np.argmax(scores))
        return self.labels[best_idx]["name"], float(scores[best_idx])

    def add(self, name: str, feat: np.ndarray, face_img: np.ndarray):
        """添加特征，超限自动多样性淘汰"""
        if name not in self.meta:
            self.meta[name] = {"ids": []}

        # 超限淘汰
        if len(self.meta[name]["ids"]) >= MAX_FEATURES_PER_PERSON:
            self._evict(name)

        # 分配新 ID
        fid = self.meta["next_id"]
        self.meta["next_id"] += 1

        # 保存排查图
        person_dir = FACE_DB_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        png_path = person_dir / f"face_{fid:03d}.png"
        cv2.imwrite(str(png_path), face_img)

        # 追加到统一矩阵（存储前先归一化）
        new_row = feat.reshape(1, -1).astype(np.float32)
        new_row = self._l2_normalize(new_row)
        if self.features.shape[0] == 0:
            self.features = new_row
        else:
            self.features = np.vstack([self.features, new_row])

        self.labels.append({"name": name, "id": fid})
        self.meta[name]["ids"].append(fid)
        self._save()
        return str(png_path)

    def _evict(self, name: str):
        """删除与其他特征平均距离最远的那个（多样性策略）"""
        ids = self.meta[name]["ids"]
        if len(ids) <= 1:
            # 只剩一个也删（刚好满10个时 add 会触发）
            evict_id = ids[0]
            self._remove_feature_by_id(name, evict_id)
            return

        # id -> 矩阵行号
        id_to_idx = {label["id"]: i for i, label in enumerate(self.labels)}
        indices = [id_to_idx[i] for i in ids]
        person_feats = self.features[indices]  # (K, 512)

        dists = pairwise_distances(person_feats)
        np.fill_diagonal(dists, np.nan)
        avg_dists = np.nanmean(dists, axis=1)
        max_local = int(np.argmax(avg_dists))

        evict_id = ids[max_local]
        self._remove_feature_by_id(name, evict_id)

    def _remove_feature_by_id(self, name: str, fid: int):
        """从矩阵、labels、meta 中删除指定 ID 的特征，并删除对应 png"""
        # 删 png
        person_dir = FACE_DB_DIR / name
        png_file = person_dir / f"face_{fid:03d}.png"
        if png_file.exists():
            png_file.unlink()

        # 找到矩阵行号
        idx = None
        for i, label in enumerate(self.labels):
            if label["id"] == fid:
                idx = i
                break
        if idx is None:
            return

        # 删矩阵行 + label
        self.features = np.delete(self.features, idx, axis=0)
        self.labels.pop(idx)

        # 更新 meta
        self.meta[name]["ids"].remove(fid)
        if not self.meta[name]["ids"]:
            del self.meta[name]

    def remove_by_id(self, fid: int):
        """按全局 ID 删除单个特征"""
        # 先找到 name
        name = None
        for n, info in self.meta.items():
            if n == "next_id":
                continue
            if fid in info.get("ids", []):
                name = n
                break
        if name is None:
            return False, f"ID {fid} 不存在"
        self._remove_feature_by_id(name, fid)
        self._save()
        return True, f"已删除 ID {fid} (属于 {name})"

    def remove_by_name(self, name: str):
        """按人名删除所有特征"""
        if name not in self.meta or name == "next_id":
            return False, f"人名 '{name}' 不存在"
        ids = list(self.meta[name]["ids"])
        for fid in ids:
            self._remove_feature_by_id(name, fid)
        self._save()
        return True, f"已删除 {name} 的所有特征（共 {len(ids)} 个）"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def recognize(image_path: str, auto_add: bool = False, add_threshold: float = 0.65):
    img, faces = detect_faces(image_path)
    if img is None:
        return {"success": False, "error": "无法读取图片", "faces_found": 0, "matches": []}

    if len(faces) == 0:
        return {"success": True, "faces_found": 0, "matches": [], "note": "未检测到人脸"}

    db = FeatureDB()
    matches = []

    for i, face_array in enumerate(faces):
        feat = extract_feature(img, face_array)
        best_name, best_score = db.search(feat)

        if best_score < SIMILARITY_THRESHOLD:
            best_name = "unknown"

        x, y, w, h = face_array[0], face_array[1], face_array[2], face_array[3]
        matches.append({
            "face_index": i,
            "name": best_name,
            "confidence": round(best_score, 3),
            "bbox": [int(x), int(y), int(w), int(h)],
            "detection_conf": round(float(face_array[14]), 3),
        })

        # 自动入库
        if auto_add and best_name != "unknown" and best_score >= add_threshold:
            face_img = get_face_roi(img, face_array)
            # 查重：与库中该人已有特征的最大相似度
            if best_name in db.meta and db.meta[best_name].get("ids"):
                id_to_idx = {label["id"]: j for j, label in enumerate(db.labels)}
                existing_ids = db.meta[best_name]["ids"]
                existing_idx = [id_to_idx[k] for k in existing_ids]
                existing_feats = db.features[existing_idx]
                q = feat.astype(np.float32).flatten()
                q = q / np.linalg.norm(q)
                dup_scores = existing_feats @ q
                max_dup = float(np.max(dup_scores))
            else:
                max_dup = 0.0

            if max_dup < 0.95:
                db.add(best_name, feat, face_img)
                matches[-1]["auto_added"] = True
            else:
                matches[-1]["auto_added"] = False
                matches[-1]["auto_add_skip_reason"] = "duplicate"

    return {
        "success": True,
        "faces_found": len(faces),
        "matches": matches,
    }


def add_face(image_path: str, name: str):
    img, faces = detect_faces(image_path)
    if img is None:
        return {"success": False, "error": "无法读取图片"}
    if len(faces) == 0:
        return {"success": False, "error": "未检测到人脸"}
    if len(faces) > 1:
        return {"success": False, "error": f"检测到 {len(faces)} 张人脸，请提供只有一张人脸的图片"}

    face_array = faces[0]
    feat = extract_feature(img, face_array)
    face_img = get_face_roi(img, face_array)

    db = FeatureDB()
    out_path = db.add(name, feat, face_img)

    return {
        "success": True,
        "name": name,
        "saved_to": out_path,
        "note": "已保存到统一特征矩阵",
    }


def remove_feature(by_id: int = None, by_name: str = None):
    db = FeatureDB()
    if by_id is not None:
        ok, msg = db.remove_by_id(by_id)
    elif by_name is not None:
        ok, msg = db.remove_by_name(by_name)
    else:
        return {"success": False, "error": "请指定 --id 或 --name"}
    return {"success": ok, "message": msg}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地人脸识别（YuNet + SFace，统一矩阵版）")
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("recognize", help="识别图片中的人脸")
    rec.add_argument("--image", required=True, help="图片路径")
    rec.add_argument("--auto-add", action="store_true", help="识别成功且置信度>=0.65时自动入库（去重阈值0.95）")

    add = sub.add_parser("add", help="手动添加人脸特征到特征库（每人最多10个）")
    add.add_argument("--image", required=True, help="图片路径")
    add.add_argument("--name", required=True, help="人名")

    rm = sub.add_parser("remove", help="删除特征（按 ID 或按人名）")
    rm_group = rm.add_mutually_exclusive_group(required=True)
    rm_group.add_argument("--id", type=int, help="全局特征 ID（查看 labels.json）")
    rm_group.add_argument("--name", help="人名，删除该人所有特征")

    args = parser.parse_args()

    if args.command == "recognize":
        result = recognize(args.image, auto_add=args.auto_add)
    elif args.command == "add":
        result = add_face(args.image, args.name)
    elif args.command == "remove":
        result = remove_feature(by_id=args.id, by_name=args.name)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))
