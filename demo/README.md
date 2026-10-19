# FP32演示：当前副本的准备状态

**本目录已复制程序和元数据，但没有模型本体及固定PNG，不应直接称为“拿到即能运行”。** 原项目Mac断网验收是历史记录，详见[原README](portable_demo/README.md)。当前缺失清单：[required_runtime_assets.json](required_runtime_assets.json)。

需要从原第七阶段便携包复制而不是重导出：

- `portable_demo/model/fp32.onnx`，6,128,891字节，SHA256 `8592824c3b716d672faa1089235d0a85dca5fc5057a7b5bcaf5571cd5fcfcf33`。
- [样例来源清单](portable_demo/samples/provenance.json)指定的10张PNG，逐一按既有SHA校验；不要裁切报告截图冒充原验证样例。

复制后执行 `python3 checks/verify_submission.py --root . --require-demo-assets`（在新仓库根目录）。此命令只校验字节，不执行模型。随后按实际目标平台安装运行时，在**新目录**验证图像预测与参考概率、损坏输入、拒绝覆盖和断网；新验收结果另存，不覆盖原历史JSON。

## 运行方式（模型和样例齐全、环境准备好之后）

原实测环境为Mac arm64。历史锁定文件不是所有平台通用保证。

```sh
cd demo/portable_demo
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.lock.txt
.venv/bin/python predict.py
.venv/bin/python app.py --port 8767
```

浏览器访问 `http://127.0.0.1:8767`。安装依赖可能需要网络；已安装依赖后的离线推理，不等于全离线安装。上传PNG/JPEG不代表能诊断任意医学图像；模型仅针对九类组织图像块，也没有自动域外识别或患者诊断能力。

## 新仓库发布要求

模型与样例可进入新库或其Release，保留相对路径和元数据校验。不要仅引用原私有库下载链接，否则老师即使有新库权限也可能缺少运行资产。原始全数据和其他14份大权重无需为默认演示迁移；完整再现需求另列资源授权说明。
