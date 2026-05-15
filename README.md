# 本地人脸识别 (face-recognition)

基于 OpenCV YuNet + SFace 的本地人脸识别插件，支持自动检测图片中的人脸并与本地特征库比对身份。采用 "Plugin 自动预处理 + Bundled Skill 指导回复"的绑定架构，实现收到照片秒认人。

## 特性

- **无需调用外部 API**：全部在本地运行，仅依赖 OpenCV DNN + numpy
- **自动检测**：通过 `pre_llm_call` hook 自动识别用户发送的图片
- **向量化搜索**：统一特征矩阵 + numpy 点乘，万人规模下搜索仍在毫秒级
- **多样性管理**：每人最多 10 个特征，超限时自动删除异常值
- **自动入库**：识别成功且置信度达标时自动追加特征，保持识别准确度

## 目录结构

```
~/.hermes/plugins/face-recognition/
├── plugin.yaml                      # 插件 manifest
├── __init__.py                      # pre_llm_call hook 实现
├── SKILL.md                         # Bundled skill 文档
├── scripts/
│   └── face_recognize.py            # 主脚本（检测、识别、入库、删除）
├── models/
│   ├── face_detection_yunet_2023mar.onnx      # YuNet 检测模型 (~227KB)
│   └── face_recognition_sface_2021dec.onnx    # SFace 识别模型 (~39MB)
├── features.npy                     # 统一特征矩阵 (N, 128)
├── labels.json                      # 特征 ID 索引
├── meta.json                        # 人物元数据
├── faces/                           # 排查小图 (128×128 PNG)
│   └── {name}/
│       └── face_{id:03d}.png
└── references/
    ├── face-recognition-architecture.md   # 架构、测试、性能
    ├── plugin-integration.md              # Plugin 绑定方案
    └── model-download.md                  # 模型下载方法
```

## 安装

### 推荐方式（通过 Hermes CLI）

```bash
hermes plugins install https://github.com/seamusmore/hermes-face-recognition.git
```

### 手动安装

```bash
# 1. 克隆到用户级插件目录
git clone https://github.com/seamusmore/hermes-face-recognition.git \
  ~/.hermes/plugins/face-recognition
```

### 启用插件
hermes plugins enable face-recognition

### 安装依赖

```bash
pip install opencv-python numpy
```

### 模型下载

```bash
# 创建模型目录
mkdir -p ~/.hermes/plugins/face-recognition/models

# 下载 YuNet
curl -L -o ~/.hermes/plugins/face-recognition/models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# SFace 较大，建议直接从 OpenCV Zoo 下载或通过镜像站
# 详见 references/model-download.md
```

### 插件生效
重启 gateway 后生效。


## 使用方式

### CLI 直接调用

```bash
# 识别
cd ~/.hermes/plugins/face-recognition
python3 scripts/face_recognize.py recognize --image photo.jpg

# 识别 + 自动入库（信任度 ≥ 0.65 时自动添加特征）
cd ~/.hermes/plugins/face-recognition
python3 scripts/face_recognize.py recognize --image photo.jpg --auto-add

# 手动添加
cd ~/.hermes/plugins/face-recognition
python3 scripts/face_recognize.py add --image photo.jpg --name alice

# 删除（按 ID 或按人名）
cd ~/.hermes/plugins/face-recognition
python3 scripts/face_recognize.py remove --id 4
python3 scripts/face_recognize.py remove --name alice
```

### Plugin 自动模式

启用插件后，用户每次发送图片，系统会自动：
1. 解析图片路径
2. 运行人脸识别
3. 将结果注入当前对话上下文
4. LLM 根据识别结果生成回复

无需手动调用任何命令。

## 阈值参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 检测阈值 | **0.9** | YuNet 检测器信任度过滤阈值，**OpenCV 官方默认值**。低于此值的检测结果会被当作无人脸处理。 |
| 识别阈值 | **0.380** | SFace cosine similarity 识别阈值。信任度 ≥ 0.380 才认为是已知人物；低于此值则返回 `unknown`。原值 0.363，已调高。 |
| 入库阈值 | **0.65** | 自动入库时的最低信任度。只有识别结果 ≥ 0.65 才会触发自动入库。 |
| 去重阈值 | **0.95** | 自动入库前的查重阈值。与库中已有特征的最大相似度 ≥ 0.95 则跳过，避免重复入库。 |
| 最多特征数 | **10** | 每个人在库中最多保存 10 个特征向量，超限时自动删除最异常的特征。 |

> 注意：YuNet 检测使用默认阈值 0.9，检测置信度不达标的照片会报"未检测到人脸"。
> 若遇到此情况，请换一张正面清晰、光线充足的照片。

## 参考文档

- [SKILL.md](SKILL.md) — 技能使用文档（给 LLM 的规则约束）
- [references/face-recognition-architecture.md](references/face-recognition-architecture.md) — 架构记录、测试结果、性能基准
- [references/plugin-integration.md](references/plugin-integration.md) — Plugin 绑定方案
- [references/model-download.md](references/model-download.md) — 模型下载方法

## 开源许可

MIT License
