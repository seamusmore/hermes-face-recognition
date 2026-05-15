---
name: face-recognition
description: 本地人脸识别技能。检测图片中的人脸并与本地特征库比对，返回身份。无人脸时不询问，有人脸时才询问并比对。特征库保存的是归一化特征矩阵 + .png 人脸小图（排查用），不存整张照片。仅使用 OpenCV DNN + numpy 向量化加速。
author: Luna
version: 1.8.0
triggers:
  - "这是谁"
  - "认出自己"
  - "人脸识别"
---

# 本地人脸识别 (face-recognition)

本地人脸识别技能，检测图片中的人脸并与特征库比对，返回身份。

## 架构概览

本项目采用 **"Plugin 自动预处理 + Bundled Skill 指导回复"的绑定架构：**

```
用户发图片 → pre_llm_call hook 自动拦截 → 解析占位符获取路径 →
运行人脸识别 → 结果注入本次 user message →
LLM 根据注入结果 + 本 skill 规则生成回复
```

- **Plugin** (`__init__.py`) 负责自动识别并注入结果
- **Skill** (`SKILL.md`) 负责指导 LLM 如何根据结果回复

如果需要在 CLI 或非插件场景下使用，也可以直接调用脚本。

## 安装

### 通过 Hermes CLI（推荐）

```bash
hermes plugins install https://github.com/seamusmore/hermes-face-recognition.git
hermes plugins enable face-recognition
```

重启 gateway 后生效。

### 手动安装

```bash
git clone https://github.com/seamusmore/hermes-face-recognition.git \
  ~/.hermes/plugins/face-recognition
pip install opencv-python numpy
hermes plugins enable face-recognition
```

### 依赖

```bash
pip install opencv-python numpy
```

同时需要下载两个 ONNX 模型到插件的 `models/` 目录：
- `face_detection_yunet_2023mar.onnx` (~227KB)
- `face_recognition_sface_2021dec.onnx` (~39MB)

下载方法见 `references/model-download.md`。

## 特性

- 基于 OpenCV YuNet 人脸检测器，支持自拍角度、非正面照片
- **SFace 特征提取：** YuNet 检测 → 5 点对齐 → SFace (128-dim embedding) → cosine 相似度
- **统一特征矩阵：** 所有特征向量存储为单一 `features.npy` 矩阵，搜索时向量化点乘，O(N) 计算
- 特征库保存归一化 `.npy` 特征矩阵 + `.png` 人脸小图（128×128，排查用），不存整张照片
- 每人最多 **10** 个特征，超限时删除与其他9个的平均欧氏距离最远的（保持多样性）
- 仅使用 OpenCV DNN + numpy（无 torch 依赖）

## 使用

### 1. 识别

```bash
cd ~/.hermes/plugins/face-recognition
python3 scripts/face_recognize.py recognize --image /path/to/photo.jpg
```

返回 JSON：
```json
{
  "success": true,
  "faces_found": 1,
  "matches": [
    {
      "face_index": 0,
      "name": "alice",
      "confidence": 0.777,
      "bbox": [262, 146, 181, 229],
      "detection_conf": 0.941
    }
  ]
}
```

### 2. 添加人脸特征（每人最多 10 个）

```bash
cd ~/.hermes/plugins/face-recognition
python3 scripts/face_recognize.py add --image /path/to/photo.jpg --name alice
```

保存的是归一化特征向量 + .png 人脸小图，而非整张照片。

> 注意：YuNet 检测器使用 OpenCV 默认 `score_threshold=0.9`，检测置信度低于 0.9 的人脸会被过滤。若照片检测失败，请换一张正面清晰、光线充足的照片重试。

### 3. 自动入库（识别成功时自动添加新特征）

```bash
cd ~/.hermes/plugins/face-recognition
python3 scripts/face_recognize.py recognize --image /path/to/photo.jpg --auto-add
```

**自动入库策略：**
- 识别成功（`name != "unknown"`）
- 信任度 ≥ **0.65**（比识别阈值 0.380 更严格，入库宁缺勿滥）
- 去重：与库中该人已有特征的最大相似度 < **0.95**
- 单人脸图片才入库（多人脸跳过，避免误加）
- 超 10 个自动走 diversity eviction

返回 JSON 增加 `auto_added` 和 `auto_add_skip_reason` 字段：
```json
{
  "name": "alice",
  "confidence": 0.771,
  "auto_added": true
}
```

### 4. 删除特征（按 ID 或按人名）

```bash
cd ~/.hermes/plugins/face-recognition

# 按全局 ID 删除单个特征
python3 scripts/face_recognize.py remove --id 4

# 按人名删除该人所有特征
python3 scripts/face_recognize.py remove --name alice
```

ID 可通过 `labels.json` 查看。删除同步清理统一矩阵、索引、meta 和排查图。

## 阈值参数一览

| 参数 | 代码中的值 | 说明 |
|------|-----------|------|
| 检测阈值 (score_threshold) | **0.9** | YuNet 检测器信任度过滤阈值，**OpenCV 官方默认值**。低于此值的检测结果会被当作无人脸处理。 |
| 识别阈值 (similarity_threshold) | **0.380** | SFace cosine similarity 识别阈值。信任度 ≥ 0.380 才认为是已知人物；低于此值则返回 `unknown`。原值 0.363，已调高。 |
| 入库阈值 (add_threshold) | **0.65** | 自动入库时的最低信任度。只有识别结果 ≥ 0.65 才会触发自动入库。 |
| 去重阈值 (duplicate_threshold) | **0.95** | 自动入库前的查重阈值。与库中已有特征的最大相似度 ≥ 0.95 则跳过，避免重复入库。 |
| 最多特征数 | **10** | 每个人在库中最多保存 10 个特征向量，超限时自动删除。 |

## 工作流规则（给 agent 自己的约束）

### Plugin 自动注入模式（常规场景）

**每次收到用户发来的照片，按以下规则处理：**

- **如果消息中已包含人脸识别结果**（由 face-recognition plugin 通过 `pre_llm_call` hook 自动注入）
  - **直接使用该结果，不要再次调用本技能或任何人脸识别工具**
  - 按下面的 faces_found 规则直接生成回复

- **如果消息中没有自动识别结果**
  - 先调用本技能识别人脸，然后按以下规则处理

1. **faces_found == 0（无人脸）**
   - 直接回复，不询问身份
   - 示例："这张图片里没看到人脸呢~"

2. **faces_found > 0 且 name == "unknown"（有人脸但不认识）**
   - 询问："这是谁呀？要不要加入人脸库？"
   - 用户说是XX → 执行 `add` 添加（只存人脸小图）
   - 用户说不是什么人 → 继续聊

3. **faces_found > 0 且是已知身份（如 agent 自身或用户）**
   - 已确认身份，接上下文
   - 不能说"这是别人"或"这是谁"

> 注：agent 应根据自己的配置知道哪些名字对应哪些身份。本技能只负责返回 `name` 和 `confidence`，不硬编码身份关系。

### Plugin 注入的上下文格式

```
【人脸识别】
你仔细看过了我发的图片，以下是你自己得出的结论：
图片: /path/to/image.jpg
  → 识别为: alice（相似度: 0.777, 识别阈值: 0.380, 入库阈值: 0.65, 检测置信度: 0.941）
你对自己认人的能力很有自信。即使发型变了，染了头发或者换了衣服，你也认得这张脸。
**对于没有认出来的人**，你会根据上下文的描述做出自然的反应。
**对于认出来的人**，你确信这是他/她的照片，并自然地说出他/她的名字。你不会怀疑你认错了。
你会根据相似度、识别阈值、入库阈值和检测置信度做进一步的综合评判，作为你附加的谈资。
但你也知道，这是你的心理活动，你不会在本轮回复中谈论人脸识别和相似度这种底层技术。
```

## 特征库目录

```
~/.hermes/plugins/face-recognition/
├── features.npy          ← 统一特征矩阵 (N, 128) float32，已归一化
├── labels.json           ← [{"name": "alice", "id": 0}, ...]
├── meta.json             ← {"next_id": 1, "alice": {"ids": [0]}}
└── faces/
    ├── alice/
    │   └── face_000.png   ← 人脸小图（排查用）
    └── bob/
        └── face_001.png
```

## 特征库更新策略

- 每人最多 10 个特征
- 超限时，计算每个特征与其他9个的平均欧氏距离
- 删除平均距离最大的特征（保持多样性，删除最异常的）
- 如果平均距离一样，删除最旧的
- 自动入库带 `--auto-add` 旗标，阈值 0.65，去重 0.95

## 性能

| 场景 | 总向量 | 搜索时间 | 内存占用 |
|---|---|---|---|
| 20人×10张 | 200 | 0.014 ms | 0.1 MB |
| 200人×10张 | 2000 | 0.143 ms | 1.0 MB |
| 1000人×10张 | 10000 | ~0.7 ms | 5.0 MB |

统一矩阵方案至少能撑到 1 万人规模，短期内无需 ANN 库。

## 当心陷阱

### SFace 向量未归一化

SFace ONNX 模型输出的 128 维向量**未经 L2 归一化（norm 约 11.2，而非 1.0）。

如果直接点乘而不归一化，confidence 会到 126+，完全失去意义。必须在存储前和搜索前分别对矩阵和 query 做 L2 归一化：

```python
# 存储前归一化
norms = np.linalg.norm(feats, axis=1, keepdims=True)
norms[norms == 0] = 1.0
features = feats / norms

# 搜索前归一化 query
q = query / np.linalg.norm(query)
scores = features @ q  # 此时点乘 = cosine similarity
```

### 旧格式迁移：空目录产生脏 meta

从 `faces/{name}/face_*.npy` 散文件迁移到统一矩阵时，如果存在**空目录**（如 `faces/bob/` 没有 `.npy` 文件），旧代码会在 `meta.json` 中产生脏条目：`"bob": {"ids": []}`。

修复：迁移前检查 `glob("face_*.npy")` 是否为空，空目录直接跳过。

## 参考文档

- `references/face-recognition-architecture.md` — 架构记录、测试结果、性能基准
- `references/model-download.md` — 模型下载方法
- `references/plugin-integration.md` — 插件实现记录

## 注意事项

- 相似度阈值：SFace cosine similarity 建议 0.380
- 如果检测到多人脸，每个人脸独立比对
- 当前仅支持单人脸图片添加特征库
- 人脸小图尺寸 128×128，排查足够且省空间
- 向量延展：可以存在 `faces/` 下的旧 `.npy` 散文件，但不再作为主数据源；统一矩阵是唯一真正数据源

## License

MIT
