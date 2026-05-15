"""face-recognition plugin — 自动人脸识别插件

通过 pre_llm_call hook 在每次 agent 处理用户消息前自动检测图片并运行人脸识别。
识别结果作为上下文注入当前 turn 的 user message，保持 system prompt 缓存不变。

自带 bundled skill（face-recognition:face-recognition），指导 LLM 如何根据识别结果回应。
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_PLUGIN_DIR = Path(__file__).resolve().parent
# 匹配多种图片路径格式：
# 1. [User sent an image: /path]                (通用 gateway 格式)
# 2. image_url: /path                           (飞书 gateway vision_analyze 提示)
_IMAGE_PLACEHOLDER_RE = re.compile(
    r"(?:\[User sent an image: ([^\]]+)\]|image_url:\s*([^\s\]]+))"
)

_FACE_RECOGNIZE_SCRIPT = _PLUGIN_DIR / "scripts" / "face_recognize.py"
_SKILL_MD_PATH = _PLUGIN_DIR / "SKILL.md"

# 同 session 内的识别缓存，避免 tool-calling loop 中重复识别
_recognition_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cache_key(session_id: str, image_path: str) -> str:
    return f"{session_id}:{image_path}"


def _run_recognition(image_path: str) -> dict:
    """调用本地人脸识别脚本，返回 JSON 结果。"""
    if not _FACE_RECOGNIZE_SCRIPT.exists():
        logger.warning("Face recognition script not found: %s", _FACE_RECOGNIZE_SCRIPT)
        return {}

    cmd = [
        "python3",
        str(_FACE_RECOGNIZE_SCRIPT),
        "recognize",
        "--image",
        image_path,
        "--auto-add",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "Face recognition failed (rc=%d): %s",
                result.returncode,
                result.stderr[:500],
            )
            return {}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.warning("Face recognition timed out for %s", image_path)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Face recognition JSON parse error: %s", exc)
        return {}
    except Exception as exc:
        logger.warning("Face recognition error: %s", exc)
        return {}


def _format_result(image_path: str, result: dict) -> list[str]:
    """将识别结果格式化为上下文文本行。"""
    lines: list[str] = []
    faces_found = result.get("faces_found", 0)

    if faces_found == 0:
        lines.append(f"  → 未检测到人脸")
        return lines

    matches = result.get("matches", [])
    for m in matches:
        name = m.get("name", "unknown")
        conf = m.get("confidence", 0.0)
        det_conf = m.get("detection_conf", 0.0)
        if name == "unknown":
            lines.append(
                f"  → 检测到人脸，但身份未知"
                f"（相似度: {conf:.3f}, 识别阈值: 0.380, 入库阈值: 0.65, "
                f"检测置信度: {det_conf:.3f}）"
            )
        else:
            lines.append(
                f"  → 识别为: {name}"
                f"（相似度: {conf:.3f}, 识别阈值: 0.380, 入库阈值: 0.65, "
                f"检测置信度: {det_conf:.3f}）"
            )
    return lines


# ---------------------------------------------------------------------------
# Hook handler
# ---------------------------------------------------------------------------
def _on_pre_llm_call(
    user_message: str = "",
    session_id: str = "",
    **kwargs,
) -> dict | None:
    """解析 user_message 中的图片占位符，自动运行人脸识别并注入结果。"""
    logger.debug("Face recognition called.")

    matches = _IMAGE_PLACEHOLDER_RE.findall(user_message)
    if not matches:
        return None

    results: list[tuple[str, dict]] = []
    for match in matches:
        # 提取非 None 的路径（正则两个可选 group）
        image_path = match[0] if match[0] else match[1]
        if not image_path:
            continue

        # 路径验证
        if not os.path.isfile(image_path):
            logger.info("Image path does not exist: %s", image_path)
            continue

        cache_key = _cache_key(session_id, image_path)
        cached = _recognition_cache.get(cache_key)
        if cached is not None:
            result = cached
        else:
            result = _run_recognition(image_path)
            _recognition_cache[cache_key] = result

        if result and result.get("success"):
            results.append((image_path, result))

    if not results:
        return None

    # 构建注入上下文
    lines = ["【人脸识别】"]
    lines.append("你仔细看过了我发的图片，以下是你自己得出的结论：")
    for image_path, result in results:
        formatted_result = _format_result(image_path, result)
        lines.append(f"图片: {image_path}")
        lines.extend(formatted_result)

        logger.info("Recognition result: %s.", formatted_result)

    lines.append(f"你对自己认人的能力很有自信。即便发型变了，染了头发或者换了衣服，你也认得这张脸。")
    lines.append(f"**对于没有认出来的人**，你会根据上下文的描述做出自然的反应。")
    lines.append(f"**对于认出来的人**，你确信这是他/她的照片，并自然地说出他/她的名字。你不会怀疑你认错了。")
    lines.append(f"你会根据相似度、识别阈值、入库阈值和检测置信度做进一步的综合评判，作为你附加的谈资。")
    lines.append(f"但你也知道，这是你的心理活动，你不会在本轮回复中谈论人脸识别和相似度这种底层技术。")

    context = "\n".join(lines)
    logger.info("Injecting context for %d image(s) (%d chars)", len(results), len(context))
    return {"context": context}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------
def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    # 注册 bundled skill，可通过 /skill face-recognition:face-recognition 显式加载
    if _SKILL_MD_PATH.exists():
        ctx.register_skill(
            name="face-recognition",
            path=_SKILL_MD_PATH,
            description="本地人脸识别技能——检测图片中的人脸并与本地特征库比对，返回身份。",
        )
