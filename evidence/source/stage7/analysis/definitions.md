# 统计定义与复现

所有输入为既有CSV；逐图序号/标签跨运行一致，无新的官方测试推理。Accuracy、Macro-F1、宏OvR AUC、混淆矩阵和逐类Precision/Recall/F1核对误差≤1e-10；概率行和容差1e-5。

E−D在同一seed分别计算，三种子均值和样本SD(ddof=1)，九类F1平均差严格等于Macro-F1差。未对概率先平均作集成。

Probability NLL=−mean(log(clip(p_true,1e-12,1)))，clip数另列；Brier为mean(sum九类(p−onehot)^2)，不除以9；ECE用15个等宽top-label置信度箱[i/15,(i+1)/15)，最后含1，空箱标∅，按n/N加权绝对校准差。固定0.90/0.95/0.99报告N、错误、覆盖率和子集错误率。无校准拟合/拒识阈值选优。

验证集参与过选模，测试已查看，这些补充均事后描述，无新盲测、病理机制或患者层面CI声明。

复现：先执行reference/handoff/scripts/preflight.py --skip-bundle-manifest（旧包仅README/AI日志更新已单独审计），再analyze_saved_predictions.py --inventory ... --output 新文件.json；禁止旧evaluate。所有输出必须新路径。
