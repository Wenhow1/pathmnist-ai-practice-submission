# 来源与版本说明

原科学仓库：`Wenhow1/pathmnist-tissue-classification`（只读）。
科学证据基线：`d5e4a71ff6a9e485fe857792461387f27dc84d1d`。
本轮读取原main：`e3337732d36a8485c5f22695c9e9ab6dadc61318`，与科学证据基线不是同一个含义。
拟新仓库：`Wenhow1/pathmnist-ai-practice-submission`；云端提交SHA和Release尚未产生。

来源分三层：

1. `technical/source_snapshot/`、`evidence/source/`、`evidence/formal/`保存原代码与已有记录，按[复制来源表](evidence/source_provenance.json)追溯。文件中字面的“当前”“最新”是其历史状态。
2. `reports/`、`presentation/`、`ai_workflow/`是当前重组材料；来源编号[S]及旧技术附件[E]均指向原记录。整理后的任务摘要不当作原提示原文。
3. `checks/`、`tests_submission/`、`.github/workflows/`是本轮新增的**提交完整性检查**；本地结果与未来云端结果独立记录。不把它们算入原正式训练或历史测试数量。

[源SHA清单](reproducibility/CHECKSUMS.sha256)覆盖当前交付文件，不包含自身和动态本地检查结果。任何后续改稿或复制运行资产都应重新生成清单并记录新提交；不可修改原冻结数据、模型或实测值来迁就检查。

本轮对两个输入包分别做148、78条SHA检查，指包内清单一致性，并非逐个从GitHub重下全文。每条复制记录保存目标与原归档路径；没有证据的原患者信息、署名或版本不会补造。
