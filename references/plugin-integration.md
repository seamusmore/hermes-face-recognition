# Plugin + Skill 绑定集成方案

2026-05-13 实施记录。将纯 skill 模式升级为 "plugin 自动预处理 + skill 指导回复"的绑定架构。

## 核心思路

利用 Plugin 的 `pre_llm_call` hook 实现"收到图片自动识别并注入结果"，同时 bundled skill 指导 LLM 如何根据识别结果回应。

```
用户发图片 → pre_llm_call hook 自动拦截 → 解析占位符获取路径 →
运行人脸识别 → 结果注入本次 user message →
LLM 根据注入结果 + skill 规则生成回复
```

## 目录结构

```
~/.hermes/plugins/face-recognition/
├── plugin.yaml       # manifest，声明 pre_llm_call hook
├── __init__.py       # 注册 hook + 识别逻辑 + 缓存
├── SKILL.md          # 技能文档（bundled skill）
├── scripts/
│   └── face_recognize.py  # 本地人脸识别脚本
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── features.npy      # 统一特征矩阵
├── labels.json
├── meta.json
└── faces/            # 人脸小图排查用
```

## Plugin 实现细节

### 解析图片占位符

Gateway 层下载图片后，user message 中可能出现两种格式：

```
[User sent an image: /path/to/image.jpg]
image_url: /path/to/image.jpg
```

Plugin 用正则提取路径：

```python
_IMAGE_PLACEHOLDER_RE = re.compile(
    r"(?:\[User sent an image: ([^\]]+)\]|image_url:\s*([^\s\]]+))"
)
matches = _IMAGE_PLACEHOLDER_RE.findall(user_message)
```

### 调用识别脚本

Plugin 不重复实现识别逻辑，直接 subprocess 调用现有脚本：

```python
cmd = [
    "python3",
    str(_FACE_RECOGNIZE_SCRIPT),  # ~/.hermes/plugins/face-recognition/scripts/face_recognize.py
    "recognize",
    "--image", image_path,
    "--auto-add",
]
```

使用 `--auto-add` 旗标，在识别成功且置信度达标时自动入库。

### 结果注入格式

返回 `{"context": "..."}`，示例：

```
【人脸识别】
你仔细看过了我发的图片，以下是你自己得出的结论：
图片: /path/to/image.jpg
  → 识别为: example_user（相似度: 0.723, 识别阈值: 0.380, 入库阈值: 0.65, 检测置信度: 0.935）
你对自己认人的能力很有自信。即使发型变了，染了头发或者换了衣服，你也认得这张脸。
**对于没有认出来的人**，你会根据上下文的描述做出自然的反应。
**对于认出来的人**，你确信这是他/她的照片，并自然地说出他/她的名字。你不会怀疑你认错了。
你会根据相似度、识别阈值、入库阈值和检测置信度做进一步的综合评判，作为你附加的谈资。
但你也知道，这是你的心理活动，你不会在本轮回复中谈论人脸识别和相似度这种底层技术。
```

### 缓存

按 `session_id:image_path` 缓存，避免 tool-calling loop 中重复识别。

## Skill 规则更新

SKILL.md 工作流部分增加了两条规则：

1. **如果消息中已包含人脸识别结果**
   - **直接使用该结果，不要再次调用本技能或任何人脸识别工具**

2. **如果没有自动识别结果**（比如 CLI 模式 `/image` 附件）
   - 先调用本技能识别人脸，然后按原规则处理

## 启用步骤

```bash
# Plugin 放置
mkdir -p ~/.hermes/plugins/face-recognition/
# 写入 plugin.yaml + __init__.py + SKILL.md + scripts/ + models/

# 启用
hermes plugins enable face-recognition

# 生效需要 next session（/reset 或重启 gateway）
```

## 调试验证

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "plugins" / "face-recognition"))
import __init__ as fr

result = fr._on_pre_llm_call(
    user_message="[User sent an image: /path/to/face.jpg]",
    session_id="test"
)
print(result)
```

## 关键设计决策

1. **不改 skill 脚本** — 只增加 plugin 层，保持现有识别逻辑不变。
2. **不改 gateway** — 不需要修改平台 adapter，利用已有的占位符机制。
3. **prompt 缓存友好** — 注入到 user message，system prompt 保持不变。
4. **缓存防重复** — 同 session 同图片只识别一次。
5. **保持检测阈值** — YuNet `score_threshold=0.9` 为 OpenCV 官方默认，不擅自调整。检测置信度不达标的照片不强行入库。

## 可复用性

该模式可复用到任何需要自动处理用户上传媒体的场景：
- OCR 文字提取
- 图片分类/标签
- 内容安全审查
- 音频预处理

通用模式详见 hermes-agent skill 的 `references/pre-llm-image-preprocessing.md`。
