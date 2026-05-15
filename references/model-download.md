# 模型下载

## 模型列表

| 模型 | 文件名 | 大小 | 来源 |
|------|--------|------|------|
| YuNet | `face_detection_yunet_2023mar.onnx` | ~227KB | OpenCV Zoo |
| SFace | `face_recognition_sface_2021dec.onnx` | ~39MB | OpenCV Zoo |

## 下载方法

### GitHub 原始源

GitHub raw 下载不稳定，但多试几次能成功。

```bash
# YuNet
curl -L -o models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# SFace
curl -L -o models/face_recognition_sface_2021dec.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
```

### 镜像源

YuNet 已成功下载的镜像：
- `https://ghps.cc`

SFace 镜像尝试失败（全部超时或被拦截）。

### 用户直接传输

当 curl 全部失败时，用户可以本地 rz 传输：

```bash
# 服务器端先安装 lrzsz
sudo yum install -y lrzsz

# 然后在插件目录执行 rz，用户从本地选择文件上传
# 上传的文件放在 models/ 子目录下
cd ~/.hermes/plugins/face-recognition/models
rz
```

## 存放路径

模型文件放在插件的 `models/` 目录下：

```
~/.hermes/plugins/face-recognition/
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
└── ...
```

## 当前状态

- YuNet: ✅ 已下载（227KB）
- SFace: ✅ 已下载（36.9MB，通过 rz 从用户本地传输）
