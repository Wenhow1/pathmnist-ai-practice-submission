# FP32演示：已补齐运行资产的新副本

本目录已包含原FP32 ONNX模型、10张固定验证PNG、元数据和便携程序。11份资产均按[指定清单](required_runtime_assets.json)匹配原SHA；[复制回执](../reproducibility/runtime_asset_copy.json)记录原归档与成员路径。没有重新导出模型、重生成图像或运行官方测试。

2026-10-19在新随机目录和新Python环境验收，未安装PyTorch、torchvision、MedMNIST或scikit-learn。[20项运行检查](../reproducibility/demo_runtime_acceptance.json)覆盖10张固定样例与原CSV参考概率（绝对误差阈值1e-5）、L/RGBA输入转换、损坏图片、模型字节／标签／预处理不匹配、路径越界与拒绝覆盖。命令行由OS禁止网络，外网探测因权限被拒绝。格式转换用例属于合成格式检查。

[4项浏览器用例](../reproducibility/browser_acceptance_20261019.json)包括正确样例、固定不一致样例、损坏输入与正常上传。应用进程由OS限制为只可访问本机回环；浏览器本身没有全局断网。课堂投影及Windows/Linux尚未验收。

## 在Mac arm64运行

本轮实测Python 3.12.14。以下python3应指向Python 3.12；如果系统默认版本不同，可用python3.12建立虚拟环境。本包不含Python安装器或离线wheelhouse。

```sh
cd demo/portable_demo
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.lock.txt
.venv/bin/python predict.py
.venv/bin/python app.py --port 8767
```

打开 http://127.0.0.1:8767 。依赖安装可能需要网络；安装完成后的推理不依赖服务器、GPU、训练数据或云API。锁文件对应实测的Mac环境，不宣称跨平台验证。

默认FP32与固定模型D_seed2026保持不变。val:9082是与给定标签不一致的固定样例，未获病理专家复核。输出是病理图像块九类组织，不是患者诊断，也没有自动域外检测。

原[portable_demo说明](portable_demo/README.md)保留历史字节，其原仓库Release及平台叙述是来源记录。当前基本运行所需文件直接随新库提供，无需另申请原私有库权限。完整NPZ和其他正式权重不属于默认演示依赖。
