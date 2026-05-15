# 人脸识别架构记录

## 流水线

```
输入图片
  ↓
YuNet 检测（自拍角度支持）
  ↓
人脸对齐（5 个关键点：两眼、鼻尖、两嘴角）
  ↓
SFace 特征提取（输出 128 维 embedding）
  ↓
L2 归一化（SFace ONNX 输出未归一化，必须手动做）
  ↓
cosine 相似度比对（归一化后点乘即 cosine sim）
  ↓
返回身份和置信度
```

> ⚠️ **关键发现**：OpenCV `FaceRecognizerSF` 的 SFace ONNX 模型输出 **128-dim** 向量（不是 512），且**未做 L2 归一化**。直接使用点乘会得到 126+ 的置信度，必须先归一化。

## 模型选择

| 组件 | 模型 | 原因 |
|------|------|------|
| 检测 | YuNet (ONNX) | 支持自拍角度、非正面，比 Haar 级联好太多 |
| 识别 | SFace (ONNX) | OpenCV 官方，128 维 embedding，cosine 相似度 |
| 备选 | ArcFace | 更好的深度学习模型，但需要更大的 onnx 文件 |

## 特征库设计

- 在 `.npy` 文件中集中保存特征向量，不保存整张照片
- 同时保存 `128x128` `.png` 人脸小图，仅用于排查
- 每人最多 **10** 个特征
- 超限时删除策略：与其他 9 个特征平均距离最远的
  - 目标：保持特征空间的多样性
  - 如果距离相同，删除最旧的

## 检测器对比

| 检测器 | 自拍角度 | 非正面 | 光线变化 | 大小 |
|--------|---------|---------|----------|------|
| Haar 级联 | ❌ 差 | ❌ 差 | ❌ 差 | 小 |
| YuNet | ✅ 好 | ✅ 好 | ✅ 好 | ~227KB |
| MTCNN | ✅ 好 | ✅ 好 | ✅ 好 | 需要 torch |
| RetinaFace | ✅ 最好 | ✅ 最好 | ✅ 最好 | 需要 torch |

## 阈值参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 检测阈值 (score_threshold) | **0.9** | YuNet 官方默认值。检测置信度 < 0.9 的人脸会被当作无人脸。 |
| 识别阈值 (similarity_threshold) | **0.380** | SFace cosine similarity 识别阈值。信任度 ≥ 0.380 才认为是已知人物。原值 0.363，已调高。 |
| 入库阈值 (add_threshold) | **0.65** | 自动入库时的最低信任度。只有识别结果 ≥ 0.65 才会触发自动入库。 |
| 去重阈值 (duplicate_threshold) | **0.95** | 自动入库前的查重阈值。与库中已有特征的最大相似度 ≥ 0.95 则跳过。 |

## 测试结果（SFace 实测）

| 图片类型 | 检测结果 | 识别结果 | 置信度 |
|---------|----------|----------|--------|
| 标准像（正面） | 1 人脸 | 命中 | **1.000** ✅ |
| 全身照（侧面+远距） | 1 人脸 | 命中 | **0.777** ✅ |
| 自拍角度（延伸手臂） | 1 人脸 | 命中 | **0.742** ✅ |
| 室内侧光（新角度） | 1 人脸 | 命中 | **0.764** ✅ |
| 发型变化 | 1 人脸 | 命中 | **0.725** ✅ |
| 户外正面（户外、正面） | 1 人脸 | 命中 | **0.771** ✅ |
| 陌生人测试（黑发） | 1 人脸 | unknown | **0.322** ✅ |
| 无脸部分（无脸） | 0 人脸 | - | 不询问 ✅ |

结论：SFace + YuNet 完全可用。识别阈值 0.380，所有相关测试都过关。陌生人被正确标为 unknown（0.322 < 0.380），不会误认。

## 下载源记录

- YuNet (227KB)：`https://ghps.cc` 镜像成功
- SFace (36.9MB)：GitHub、HuggingFace、KDE 镜像全部超时/失败
  - 最终通过本地 `rz` 传输成功
  - 先执行 `sudo yum install -y lrzsz`，然后在插件目录执行 `rz`

## 统一特征矩阵架构（v1.4+）

### 为什么从散文件迁移到统一矩阵

旧实现：`faces/{name}/face_000.npy` 散文件，每次识别遍历文件夹逐个 `np.load()`。

问题：
- **IO 瓶颈**：200 个文件逐个加载 **16.83 ms**
- **计算瓶颈**：Python `for` 循环逐个算 cosine_similarity
- **无缓存**：每次 `recognize()` 都重新加载全部

新实现：统一 `features.npy` + `labels.json` + `meta.json`

| 文件 | 内容 | 大小（200 向量）|
|---|---|---|
| `features.npy` | `(N, 128)` float32，**已 L2 归一化** | ~100 KB |
| `labels.json` | `[{"name": "alice", "id": 0}, ...]` | ~1 KB |
| `meta.json` | `{"next_id": 5, "alice": {"ids": [0,1,2,3,4]}}` | ~1 KB |

搜索时：
```python
scores = features_matrix @ query_norm  # (N,) 向量化点乘
best_idx = np.argmax(scores)
```

### 性能实测

| 场景 | 总向量 | 矩阵搜索 | 散文件 IO | 内存占用 |
|---|---|---|---|---|
| 20人×10张 | 200 | **0.014 ms** | 16.83 ms | 0.1 MB |
| 200人×10张 | 2000 | **0.143 ms** | - | 1.0 MB |
| 1000人×10张 | 10000 | **~0.7 ms** | - | 5.0 MB |

结论：统一矩阵至少能撑到 **1 万人** 规模，短期内无需 ANN 库。

### 向后兼容

启动时检测统一矩阵是否存在。不存在则自动从 `faces/*/*.npy` 旧格式迁移，并执行 L2 归一化。旧 `.npy` 散文件保留在 `faces/` 下，但不再作为主数据源。

## 自动入库与删除策略

### 自动入库（`--auto-add`）

```bash
python3 scripts/face_recognize.py recognize --image photo.jpg --auto-add
```

触发条件：
1. 识别成功（`name != "unknown"`）
2. 置信度 ≥ **0.65**（比识别阈值 0.380 更严格）
3. 去重：与库中该人已有特征的最大相似度 < **0.95**
4. 单人脸图片（多人脸跳过，避免误加）

### 手动删除

```bash
# 按全局 ID 删单个特征
python3 scripts/face_recognize.py remove --id 4

# 按人名清空该人所有特征
python3 scripts/face_recognize.py remove --name alice
```

删除同步清理：统一矩阵行、labels.json 条目、meta.json 元数据、`faces/{name}/face_{id}.png` 排查图。

## 关键坑点

### 坑 1：SFace 输出维度不是 512 是 128

OpenCV 文档写 512-dim，但 ONNX 模型 `face_recognition_sface_2021dec.onnx` 实际输出 `(128,)` float32。

### 坑 2：SFace 输出未归一化

`cv2.FaceRecognizerSF.feature()` 返回的向量 norm ≈ 11.24，**不是 unit vector**。直接点乘会得到 126+ 的置信度。必须显式 L2 归一化：

```python
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0
return mat / norms
```

### 坑 3：空目录污染 meta

`faces/<name>/` 空目录会导致 `meta.json` 生成 `"<name>": {"ids": []}`。迁移逻辑必须过滤无 `.npy` 文件的目录。
