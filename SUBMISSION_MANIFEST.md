# 提交内容与完成边界

| 内容 | 当前状态 | 验收依据 |
|---|---|---|
| AI工程实践主报告＋三附件 | 已完成当前文稿、可编辑Word与PDF | 文件清单、导出页面检查 |
| 同版PPT、PDF和讲稿 | 已完成30页版本 | 备注、页序、隐藏附录及包结构检查 |
| 必要代码与轻量证据 | 已复制，来源逐项记录 | source_provenance与SHA |
| 新提交完整性脚本／合成单元测试 | 已准备并本地执行 | local_check.json与unit_tests.log |
| 原模型／固定图像在本新目录 | **缺模型1个、PNG10个** | demo/required_runtime_assets.json |
| 新仓库、推送和Release | **未创建／未发布** | 等待实际仓库URL、commit与资产校验 |
| GitHub Actions | 配置存在，**云端未运行** | 必须有新的run URL和真实结论后更新 |
| 教师账号读取／下载 | 未授权验收 | 所有者可读不能替代老师可读 |
| 正式模型实验 | 沿用历史，不新增 | d5e4a71固定证据 |

完整精确状态由[submission_status.json](reproducibility/submission_status.json)记录。这里的提交材料版本不是新增实验版本。
