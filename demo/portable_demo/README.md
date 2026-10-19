# PathMNIST 独立 FP32 离线演示

冻结模型 D_seed2026，对病理RGB图像块预测九类组织。没有患者诊断或自动域外检测功能。失败 INT8 不在此包中，也不提供自动切换。

已实测：macOS 15.4.1 arm64，Python 3.12.14，ONNX Runtime 1.31.0，CPU 单线程。搬到随机新目录、无 PyTorch/数据集/训练代码，并由操作系统禁止外部网络后，CLI 与 Chrome 界面验收通过。Windows/Linux 尚未实测；现场投影与视频播放仍需自行检查。锁定文件针对实测平台，其他平台不能宣称兼容。

先在联网时安装运行时（包不含 Python 安装器或 wheelhouse）：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.lock.txt
```

依赖安装完后即可断网：

```sh
.venv/bin/python predict.py
.venv/bin/python predict.py samples/val_9082.png
.venv/bin/python app.py --port 8767
```

浏览器打开 http://127.0.0.1:8767 。默认 CLI 是 FP32 并预测固定正确样例 val:0，`--output 新文件.json` 会拒绝覆盖已有文件。仅本机回环地址提供界面，不需要云 API、GPU 或远程服务器。

上传 PNG/JPEG，灰度/RGBA转换RGB；最大20MB、1600万像素。固定样例来自验证集，provenance.json保留原CSV九类概率、原数组像素回读核验及来源SHA。val:9082与数据集给定标签不一致，未经病理专家复核；现场预测仍执行模型，并非读取参考答案。

模型元数据与SHA、冻结ID、checkpointSHA、九类顺序、RGB/BILINEAR/128/float32/NCHW、ImageNet均值标准差一并验证，任何不匹配拒绝。禁止绝对路径、父目录及逃出模型目录的软链接。

时间分别报告会话创建、CPU前向、含解码预处理的处理耗时；这些单次演示时间不是正式PyTorch基准中位/P95，不应混用。

离线推理不等于完全离线安装。此包不含全部15份权重、原始数据或INT8模型；正式大文件另见原项目私有GitHub Release。SHA256SUMS由本轮新打包工具生成，先校验再运行。
