# AI辅助医学图像分类的工程实践

**PathMNIST · 研究生工程实践第一阶段 · 教师提交材料 v1**

**独立私有教师仓库已创建并上传**：[Wenhow1/pathmnist-ai-practice-submission](https://github.com/Wenhow1/pathmnist-ai-practice-submission)。原仓库保持只读，本库不是Fork。2026-10-19发布接手补齐了原运行资产，并取得新副本验收和真实CI记录。精确提交、Release和CI见[云端回执](reproducibility/cloud_verification.json)。

## 从这里阅读

| 目的 | 入口 |
|---|---|
| 先看课程主线 | [教师阅读与运行入口](00_教师阅读入口.md) |
| 看课堂汇报 | [30页PPT](presentation/PathMNIST_AI工程实践汇报_教师提交版.pptx) · [全页PDF](presentation/PathMNIST_AI工程实践汇报_全页预览.pdf) |
| 看完整过程 | [AI工程实践主报告PDF](reports/01_AI工程实践主报告.pdf) · [Word](reports/01_AI工程实践主报告.docx) |
| 查数据 | [数据集探索笔记](reports/02_数据集探索笔记.pdf) |
| 查交互 | [AI使用日志与交互案例](reports/03_AI使用日志与交互案例.pdf) · [原始来源索引](reference/来源编号与阅读索引.md) |
| 查模型和结果 | [模型与工程评估报告](reports/04_模型与工程评估报告.pdf) · [正式源表](evidence/all_runs.csv) |
| 准备答辩 | [同版逐页讲稿与图示说明](presentation/逐页讲稿与答疑.md) |
| 看代码／复核 | [原代码快照](technical/source_snapshot/) · [安全核查说明](reproducibility/运行命令.md) |
| 运行离线演示 | [演示状态与准备步骤](demo/README.md)（模型和10张PNG已补齐并验收） |
| 查发布过程 | [发布接手记录](reproducibility/发布接手记录.md) · [收到时的交接要求](codex_handoff/HANDOFF.md) |

## 项目与证据版本

实验使用PathMNIST官方固定划分；A—E五组各三个种子，共15次正式训练。技术结果仍固定原科学证据提交 `d5e4a71ff6a9e485fe857792461387f27dc84d1d`。本轮只重组课程材料和添加提交完整性检查，不重新训练、运行官方测试推理或更换冻结的D_seed2026。

原仓库是 `Wenhow1/pathmnist-tissue-classification`，本轮只读所见main为 `e3337732d36a8485c5f22695c9e9ab6dadc61318`。本目录引用的来源主要从用户已提供且校验一致的归档复制，并记录[复制来源与SHA](evidence/source_provenance.json)；不把档案内自述全部提升成本轮远程重新核验。

## 交付状态必须分开

- 文稿、PPT、来源快照、提交检查脚本和CI配置：已在当前目录准备。
- 当前可执行的本地核查：只检查保存文件和确定性算术，不导入训练框架、不推理模型。
- 新副本运行：独立Mac arm64环境，命令行由操作系统禁止网络，20项检查通过；浏览器另外完成4项用例。见[运行回执](reproducibility/demo_runtime_acceptance.json)和[界面回执](reproducibility/browser_acceptance_20261019.json)。
- 私有仓库和推送已完成，真实Actions已通过；只检查文件、哈希与保存结果算术。Release的发布提交、附件及哈希单独保存在云端回执。教师账号尚未提供，未邀请或验收教师访问。
- 本目录直接包含 `demo/portable_demo/model/fp32.onnx` 与10张固定验证PNG，基本演示不依赖原仓库权限。安装依赖仍需网络或另备离线安装包。

PPT共30页：1—25页正文、26页结束／答疑索引、27—30页隐藏附录；本轮不强制十分钟。旧21／23／25页手册不能套用新页码。

学号、实际提交日期和每位成员贡献由本人填写。没有实施K折、QAT或Claude Code；未记录的专用Skill、无AI开发对照、病理专家或临床验证不编造。
