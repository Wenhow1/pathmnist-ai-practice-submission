# 21页 PPT 模块—箭头—代码对照

目标：用户请求包 reference/PathMNIST_浙大模板_十分钟答辩版.pptx，SHA256 `dc71be849eaf5d52fdbca04e8cce9ed32c195be1f0f177e7b632d45a812b0be6`。它不是 GitHub 旧20页、22页空白模板，也不是刚上传的另一份21页排版。第1—13页主讲预算560秒（9分20秒），第14页结束约5秒，15—21页附录隐藏；均非实测用时。

本轮直接查看全部21页渲染图，未OCR；对象、媒体关系与裁切记录同时核对。模块名称按画面，箭头包括图中的线、文字箭头及明确标注的隐含/编号控制关系，装饰与导航不当科学数据流。源代码版本一律d5e4a71；每个源码文件SHA和函数完整行号见《源码函数行号索引.csv》。维度为按实现或既有机制记录推导，未新运行测量。

第三方结构边界：本地未找到与正式 torchvision 0.29.0+cu130 配套归档的 resnet.py / mobilenetv3.py，因此不声称逐层核验了该版本库源码。图中残差、倒残差、深度卷积、部分SE与GAP按既有G01/当前PPT机制记录解释；确定的项目分类头是 fc=Linear(fc.in_features,9) 与 classifier[-1]=Linear(classifier[-1].in_features,9)。没有证据时不写死内部通道数、重复块数或512/1024等输入宽度。它们需要第三方源码才能升级为逐层精确拓扑，不影响本轮对现有简图和真实项目调用的说明。

共同术语：logits是网络原始九类分数；softmax用于本项目评估概率；交叉熵用于训练；验证Macro-F1用于控制选权重。ImageNet均值/标准差只是固定数值归一化；量化校准估计数值范围，概率可靠性分析则比较置信度与正确率，二者无关。统计/表格模块不学习参数。网络各层可学习参数只在训练中经反向和AdamW更新，验证/测试/演示不更新。

## 第1页：封面

### P1　研究题目与范围；中部标题

- **功能与输入：**读取现有项目任务
- **输出与维度：**九类图像块分类、2×2主对照＋增强消融
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：只用于介绍
- **采用原因与证据层：**历史任务定位，不是新网络贡献
- **来源/函数/行号/版本：**docs/方法与实验协议.md；stage3/matrix.json

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第2页：任务：由一张图像块判断九类组织

### P2a　训练集 / 验证集 / 另一中心测试集；左侧三行

- **功能与输入：**官方 train/val/test 图像和0—8标签，数组索引定位图像
- **输出与维度：**89996/10004/7180 张；原图128×128×3 uint8
- **可学习参数及更新：**数据本身无参数；只有训练图像进入梯度更新
- **使用阶段与行为：**训练/验证/测试/推理：训练更新；验证选轮次/配置；测试冻结后评价
- **采用原因与证据层：**历史官方划分；来源患者独立声明与NPZ可核验事实分开
- **来源/函数/行号/版本：**pathmnist_project/data.py::PathDataset L143–177 @d5e4a71；stage1/manifest.json

### P2b　九类组织图像；右侧3×3

- **功能与输入：**已有O01组织样例，按所附裁切记录显示
- **输出与维度：**九种组织示例，不表示患者数或9个患者
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：只作数据说明；非新推理
- **采用原因与证据层：**图像类别名称取项目标签顺序；不凭肉眼做病理诊断
- **来源/函数/行号/版本：**stage1/sample_provenance.json；stage1/class_examples.png；reference/素材裁切记录.json

### P2c　64×64由同图128缩小；底部句子

- **功能与输入：**同一split/index原始128图像
- **输出与维度：**3×H×W float，H=W=64或128
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：所有阶段均按该运行size；仅训练可随机增强
- **采用原因与证据层：**历史实现保证样本/标签对应；不等同官方独立64包
- **来源/函数/行号/版本：**pathmnist_project/data.py::transform L134–140 @d5e4a71；pathmnist_project/data.py::PathDataset L143–177 @d5e4a71

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P2-A：同张128图像 → 64输入|图像像素经双线性抗锯齿缩放；该分辨率运行的数据读取|仅更改单张图像大小；**不表示：**不产生新患者或重新划分|pathmnist_project/data.py::transform L134–140 @d5e4a71|

## 第3页：三个研究问题，五组受控实验

### P3　A—E实验表；中部五行

- **功能与输入：**两种模型、两种size、augment和3seeds
- **输出与维度：**15条预定轨迹
- **可学习参数及更新：**表格无参数；网络各自训练
- **使用阶段与行为：**训练/验证/测试/推理：全部正式训练/验证/测试；E不参与选配置
- **采用原因与证据层：**历史矩阵；共同超参数不等于每架构独立调到最优
- **来源/函数/行号/版本：**pathmnist_project/experiment.py::make_matrix L12–16 @d5e4a71；stage3/matrix.json

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P3-A/B：A → B|输入尺寸的指定对照关系；比较保存结果|同seed、同划分、共同超参数；按指定配对比较；**不表示：**不是流程箭头；A/D不能称单因素；同增强策略不保证逐图随机变换同步|pathmnist_project/experiment.py::make_matrix L12–16 @d5e4a71|
|P3-C/D：C → D|输入尺寸的指定对照关系；比较保存结果|同seed、同划分、共同超参数；按指定配对比较；**不表示：**不是流程箭头；A/D不能称单因素；同增强策略不保证逐图随机变换同步|pathmnist_project/experiment.py::make_matrix L12–16 @d5e4a71|
|P3-A/C：A → C|架构方案的指定对照关系；比较保存结果|同seed、同划分、共同超参数；按指定配对比较；**不表示：**不是流程箭头；A/D不能称单因素；同增强策略不保证逐图随机变换同步|pathmnist_project/experiment.py::make_matrix L12–16 @d5e4a71|
|P3-B/D：B → D|架构方案的指定对照关系；比较保存结果|同seed、同划分、共同超参数；按指定配对比较；**不表示：**不是流程箭头；A/D不能称单因素；同增强策略不保证逐图随机变换同步|pathmnist_project/experiment.py::make_matrix L12–16 @d5e4a71|
|P3-D/E：D → E|整套几何增强的指定对照关系；比较保存结果|同seed、同划分、共同超参数；按指定配对比较；**不表示：**不是流程箭头；A/D不能称单因素；同增强策略不保证逐图随机变换同步|pathmnist_project/experiment.py::make_matrix L12–16 @d5e4a71|

## 第4页：模型方案：经典残差网络与轻量网络

### P4-R0　RGB图像64×64或128×128；左列顶部

- **功能与输入：**缩放归一化图像
- **输出与维度：**N×3×H×W
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/data.py::transform L134–140 @d5e4a71；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-R1　卷积与池化；左列第二框

- **功能与输入：**图像张量；提取局部纹理并汇聚
- **输出与维度：**N×C₁×h₁×w₁（符号推导）
- **可学习参数及更新：**卷积等有参数；池化运算本身无
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json中的核心机制记录；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-R2　残差特征提取 / 快捷连接与主分支相加；左列第三框

- **功能与输入：**x送主分支F；捷径S对齐后相加
- **输出与维度：**F(x)+S(x)，两路形状必须相容
- **可学习参数及更新：**主分支/投影可有参数；相加操作本身无
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01既有记录；未归档该版第三方BasicBlock源码；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-R3　全局平均池化；左列第四框

- **功能与输入：**空间特征图 N×C×h×w
- **输出与维度：**N×C×1×1，再展平N×C（原理说明）
- **可学习参数及更新：**平均运算无
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01既有记录；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-R4　九类全连接层；左列末框

- **功能与输入：**上游特征；代码实际为Linear(model.fc.in_features,9)
- **输出与维度：**N×9 logits；不在此层调用softmax
- **可学习参数及更新：**最后全连接层可学习
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-M0　RGB图像64×64或128×128；右列顶部

- **功能与输入：**同任务图像张量
- **输出与维度：**N×3×H×W
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/data.py::transform L134–140 @d5e4a71；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-M1　卷积入口；右列第二框

- **功能与输入：**图像张量进入轻量特征路径
- **输出与维度：**N×C₁×h₁×w₁（符号推导）
- **可学习参数及更新：**有可学习卷积参数
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01既有记录；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-M2　轻量倒残差特征提取 / 深度卷积、通道重标定；右列第三框

- **功能与输入：**通道扩展/逐通道空间卷积/投影的机制概括；部分块使用SE通道权重
- **输出与维度：**N×C₂×h₂×w₂；若有残差需尺寸相容
- **可学习参数及更新：**卷积与SE映射有参数；不是每块都有SE或可相加
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01既有机制记录；stage7/quant_candidate_plan.json五节点含SE的fc1/fc2；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-M3　卷积整合与全局平均池化；右列第四框

- **功能与输入：**轻量特征图经卷积整合，空间平均
- **输出与维度：**N×C×1×1，再展平（符号推导）
- **可学习参数及更新：**卷积有参数，平均操作无
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01既有记录；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P4-M4　九类分类器；右列末框

- **功能与输入：**保留torchvision classifier上游，仅classifier[-1]换Linear(in_features,9)
- **输出与维度：**N×9 logits；不是声称整段classifier仅一层
- **可学习参数及更新：**末层及保留的网络分类器可学习
- **使用阶段与行为：**训练/验证/测试/推理：训练全网络AdamW更新；验证/测试/推理走前向，eval()且不反传
- **采用原因与证据层：**历史使用两种标准预训练网络；机制作用属于原理说明，不是本项目消融结论
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；pathmnist_project/engine.py::train L86–170 @d5e4a71

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P4-R0：RGB图像64×64或128×128 → 卷积与池化|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-R1：卷积与池化 → 残差特征提取 / 快捷连接与主分支相加|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-R2：残差特征提取 / 快捷连接与主分支相加 → 全局平均池化|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-R3：全局平均池化 → 九类全连接层|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-M0：RGB图像64×64或128×128 → 卷积入口|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-M1：卷积入口 → 轻量倒残差特征提取 / 深度卷积、通道重标定|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-M2：轻量倒残差特征提取 / 深度卷积、通道重标定 → 卷积整合与全局平均池化|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-M3：卷积整合与全局平均池化 → 九类分类器|首箭头为图像张量；中间为特征；进入末层为汇聚特征；训练及所有评估前向|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P4-bottom1：各自ImageNet V1预训练权重 → 替换九类输出层|初始化参数与结构改造步骤；构建模型、正式训练前|迁移既有视觉表征并匹配9类任务；**不表示：**不表示把ImageNet图片与PathMNIST混合输入|pathmnist_project/model.py::build_model L7–16 @d5e4a71|
|P4-bottom2：替换九类输出层 → 全网络微调|控制步骤；optimizer接收model.parameters()；正式训练|更新全部可学习参数；**不表示：**不是只训练新末层，也未证明全微调一定优于冻结|pathmnist_project/engine.py::train L86–170 @d5e4a71|

## 第5页：训练过程：从一批图像到一次参数更新

### P5a　预处理；上排第一框

- **功能与输入：**PIL RGB及运行size/augment
- **输出与维度：**单图3×H×W float32，批后N×3×H×W
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：只训练集随机H/V翻转及QuarterTurn；val/test关增强
- **采用原因与证据层：**历史实现：ImageNet数值归一化，不是染色归一化
- **来源/函数/行号/版本：**pathmnist_project/data.py::transform L134–140 @d5e4a71；pathmnist_project/data.py::PathDataset L143–177 @d5e4a71

### P5b　网络前向；上排第二框

- **功能与输入：**图像N×3×H×W；标签不拼入图像
- **输出与维度：**N×9 logits
- **可学习参数及更新：**有，全网络训练
- **使用阶段与行为：**训练/验证/测试/推理：train()时训练；infer()时eval且无梯度
- **采用原因与证据层：**输出未归一化九类分数
- **来源/函数/行号/版本：**pathmnist_project/engine.py::train L86–170 @d5e4a71；pathmnist_project/engine.py::infer L39–49 @d5e4a71

### P5c　交叉熵损失；上排第三框

- **功能与输入：**N×9 logits与N个整数标签y
- **输出与维度：**标量loss
- **可学习参数及更新：**损失运算无独立参数
- **使用阶段与行为：**训练/验证/测试/推理：正式训练；验证没有用标签反向更新
- **采用原因与证据层：**历史代码CrossEntropyLoss；F1不是直接训练目标
- **来源/函数/行号/版本：**pathmnist_project/engine.py::train L86–170 @d5e4a71

### P5d　反向传播 / AdamW更新全网络；上排第四框

- **功能与输入：**loss.backward得到每个可学习参数的梯度
- **输出与维度：**optimizer.step后的新参数
- **可学习参数及更新：**AdamW状态/参数随训练更新
- **使用阶段与行为：**训练/验证/测试/推理：仅训练；zero_grad在每批先调用
- **采用原因与证据层：**历史实现；不是重新生成一张图像
- **来源/函数/行号/版本：**pathmnist_project/engine.py::train L86–170 @d5e4a71

### P5e　每轮结束验证；下部左第一行

- **功能与输入：**当前模型、10004张val图像及给定标签
- **输出与维度：**九类概率、val Macro-F1/Accuracy
- **可学习参数及更新：**不更新参数；eval且inference_mode
- **使用阶段与行为：**训练/验证/测试/推理：每epoch后，推理softmax再计指标
- **采用原因与证据层：**历史控制流程，指标评价与梯度分开
- **来源/函数/行号/版本：**pathmnist_project/engine.py::infer L39–49 @d5e4a71；pathmnist_project/metrics.py::classification_metrics L10–35 @d5e4a71；pathmnist_project/engine.py::train L86–170 @d5e4a71

### P5f　保存最佳 / 连续8轮未提高早停 / 重载；下部左三行

- **功能与输入：**验证Macro-F1、best与stale计数
- **输出与维度：**best.pt及对应val记录；终止/重载决策
- **可学习参数及更新：**保存文件无新参数更新
- **使用阶段与行为：**训练/验证/测试/推理：每轮后；训练终止后再核对重载指标
- **采用原因与证据层：**改善条件是严格大于；40轮上限或patience先到先停
- **来源/函数/行号/版本：**pathmnist_project/engine.py::train L86–170 @d5e4a71

### P5g　共同设置；右下表

- **功能与输入：**batch128、lr.0003、wd.0001、余弦40
- **输出与维度：**统一训练协议
- **可学习参数及更新：**超参数，不是从报告数据拟合的新参数
- **使用阶段与行为：**训练/验证/测试/推理：A—E正式训练；smoke的16另列
- **采用原因与证据层：**batch为共同可行性试跑结果，不是精度最优证明
- **来源/函数/行号/版本：**pathmnist_project/engine.py::probe L173–215 @d5e4a71；stage3/matrix.json

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P5-A1：预处理 → 网络前向|图像张量N×3×H×W；每个训练batch|将数值图像送模型；**不表示：**不是标签流|pathmnist_project/engine.py::train L86–170 @d5e4a71|
|P5-A2：网络前向 → 交叉熵损失|N×9 logits；训练前向后|和真实类别标签共同计算loss；**不表示：**不是已校准概率，也不需先softmax|pathmnist_project/engine.py::train L86–170 @d5e4a71|
|P5-A3：交叉熵损失 → 反向传播|标量loss及计算图的求导关系；loss.backward()|计算梯度供AdamW更新；**不表示：**向右箭头是流程顺序，不表示梯度沿前向同方向穿过各层|pathmnist_project/engine.py::train L86–170 @d5e4a71|
|P5-A4：验证Macro-F1提高 → 保存最佳权重|指标比较产生的控制决策；每轮验证后|决定是否写best.pt和对应val预测；**不表示：**不是用验证标签反传|pathmnist_project/engine.py::train L86–170 @d5e4a71|
|P5-A5：连续8轮未提高 → 早停|stale计数/终止控制；每epoch尾|停止无改善的训练轨迹；**不表示：**不是8轮总训练上限|pathmnist_project/engine.py::train L86–170 @d5e4a71|
|P5-A6：训练结束 → 重载最佳权重并核对指标|best.pt模型文件；早停或40轮后|恢复最优val epoch并核对保存一致性；**不表示：**不自动使用最后轮权重|pathmnist_project/engine.py::train L86–170 @d5e4a71|
|P5-hidden-y：N个真实标签 → 交叉熵损失|整数类别标签；图中仅文字说明，没有独立标签箭头；训练|监督学习损失必须知道目标类别；**不表示：**不属于图像输入或网络中间特征|pathmnist_project/engine.py::train L86–170 @d5e4a71|

## 第6页：先验证选模，再冻结并评价另一中心

### P6a　Accuracy / Macro-F1 / 辅助指标；左列

- **功能与输入：**保存标签y与N×9概率
- **输出与维度：**Accuracy、逐类P/R/F1、宏AUC；Macro-F1=mean(F1_0..8)
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：验证/测试用于评价；推理单图无真值不计算Accuracy
- **采用原因与证据层：**历史主指标；Macro-F1不是概率可靠性保证
- **来源/函数/行号/版本：**pathmnist_project/metrics.py::classification_metrics L10–35 @d5e4a71

### P6b　测试前选择规则1—4；右列

- **功能与输入：**A—D各3个val F1及冻结前统一CPU中位延迟
- **输出与维度：**A/B/D入围→D→D_seed2026→冻结15条轨迹
- **可学习参数及更新：**控制逻辑无可学习参数
- **使用阶段与行为：**训练/验证/测试/推理：训练全部完成后、正式test生成前
- **采用原因与证据层：**0.005为实现中的工程取舍容差；CPU只用零张量
- **来源/函数/行号/版本：**pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；stage5/frozen.json

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P6-step1：A—D验证组均值 → 距最高组≤0.005的候选集|验证指标和容差规则；编号步骤所表达的控制流，无逐条画线|限定精度损失可接受的工程集合；**不表示：**不是统计等效检验|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71|
|P6-step2：候选A/B/D → CPU前向更快的D|每组3个运行中位延迟的中位数；编号步骤所表达的控制流，无逐条画线|按既定速度规则选择配置；**不表示：**不是图9展示的中位延迟组均值|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71|
|P6-step3：D内验证F1 → D_seed2026|种子验证排序；编号步骤所表达的控制流，无逐条画线|取D中最高验证F1，平分按seed；**不表示：**不是按测试分数选seed|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71|
|P6-step4：已选模型及15条轨迹 → 冻结后官方外测|模型/配置/来源哈希锁与控制决策；编号步骤所表达的控制流，无逐条画线|先固定再评价全部预定轨迹；**不表示：**不是只测D；E始终为消融|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71|

## 第7页：验证接近满分，先检查高分来自哪里

### P7a　数据与计算链条；左列四行

- **功能与输入：**历史数组/标签/推理状态/逐图指标核对
- **输出与维度：**12份A—D验证复算；精确跨集重复0
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：历史开发验证审计；本轮仅读记录
- **采用原因与证据层：**检查工程错误，不能证明患者隔离
- **来源/函数/行号/版本：**pathmnist_project/data.py::PathDataset L143–177 @d5e4a71；pathmnist_project/engine.py::infer L39–49 @d5e4a71；stage4/audit/高验证分数来源审计.md

### P7b　9032 / 849 / 91 / 排除敏感性；右列

- **功能与输入：**8方向启发式哈希与像素阈值筛查的既有候选
- **输出与维度：**9032候选对→849像素候选对→91不同val索引；B/D平均F1变化<3e−6
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：历史val敏感性描述，不删官方数据、不重训
- **采用原因与证据层：**候选不足以解释高分；不把相似图像判成同患者
- **来源/函数/行号/版本：**stage4/audit/near_duplicate_scan.json；candidate_sensitivity.json及其方法说明

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P7-schematic：候选对筛查 → 涉及91个val索引的敏感性复算|索引集合/原概率子集；页面是数字递进，未画流程线；历史val审计|说明筛查数量和指标敏感性；**不表示：**不是9032或849个患者，也不是本轮删数据|stage4/audit/candidate_sensitivity.json|

## 第8页：同中心验证近满分，另一中心表现下降

### P8　蓝橙分组柱图与验证—测试落差图；中部双面板

- **功能与输入：**all_runs各组val/test F1
- **输出与维度：**左：组均值±样本SD；右：同seed(val−test)差值汇总
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：训练后的保存结果描述
- **采用原因与证据层：**沿用O03；只测64/128，B−A有1个seed为负
- **来源/函数/行号/版本：**stage5/final/all_runs.csv；generalization_plot_provenance.json；validation_vs_external_test.png

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第9页：同为128输入，D的效果与开销更均衡

### P9　B/D三项指标及散点图；左数字、右散点

- **功能与输入：**组测试F1、parameters及冻结CPU基准
- **输出与维度：**横轴cpu_median_ms的组均值，纵轴test F1均值；误差条样本SD；参数11.18M/1.53M
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：旧PyTorch服务器CPU、threads1、batch1、50预热/500计时
- **采用原因与证据层：**当前协议效果—开销比较，非所有设备排行榜
- **来源/函数/行号/版本：**pathmnist_project/engine.py::cpu_benchmark L52–74 @d5e4a71；stage5/final/all_runs.csv；pathmnist_project/analysis.py::summarize L66–129 @d5e4a71

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第10页：增强消融在验证与外部测试中方向相反

### P10　D−E验证与测试连线；左图及右侧数字

- **功能与输入：**同seed D和E保存的F1
- **输出与维度：**val平均+.001552，test平均−.010498；线只连接同seed两个split的差值
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：保存结果的配对描述；D−E开启减关闭
- **采用原因与证据层：**整套几何增强的外部结果，不拆解三增强各自贡献
- **来源/函数/行号/版本：**stage5/final/all_runs.csv；augmentation_direction_flip.png及provenance

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P10-lines：同seed验证D−E点 → 同seed测试D−E点|同seed的两个差值的可视化连接线；结果图|看方向反转；**不表示：**不是训练时间曲线、因果通路或传播特征|stage5/final/generalization_plot_provenance.json|

## 第11页：总体准确率较高，弱类别仍需单独分析

### P11a　混淆矩阵与94.87%/0.9290/62.47%；左图、右上

- **功能与输入：**D_seed2026单模型test指标
- **输出与维度：**行真实、列预测；总n7180，class7 support421，TP263；7→5=79
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：冻结模型外测旧记录；非D组三种子汇总
- **采用原因与证据层：**总体Accuracy和等权F1需与弱类Recall同时报告
- **来源/函数/行号/版本：**stage7/evidence/formal/D_seed2026/test_metrics.json；pathmnist_project/metrics.py::classification_metrics L10–35 @d5e4a71

### P11b　test:35、test:56、test:15；右下三图

- **功能与输入：**原O07首行既有图像裁切和保存预测CSV
- **输出与维度：**35:7→5；56:7→5；15:5→5正确对照；概率见固定行查询JSON
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：仅历史test错误/对照示例展示
- **采用原因与证据层：**图像块给定标签，不作病理专家机制结论
- **来源/函数/行号/版本：**stage5/final/error_examples_provenance.json；reference/PPT11固定误例_原CSV行查询.json；reference/素材裁切记录.json

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P11-label：真实癌相关间质7 → 预测平滑肌5|混淆矩阵cell(7,5)的79个图像计数；旧test结果|表示错误方向；**不表示：**不是病变转化、生物因果或患者漏诊率|stage7/evidence/formal/D_seed2026/test_metrics.json::confusion_matrix[7][5]|

## 第12页：量化恢复了验证精度，但未获得CPU加速

### P12a　冻结权重 / FP32 ONNX / 训练后量化；上部文字流程

- **功能与输入：**相同D_seed2026权重，经eval导出ONNX并优化
- **输出与维度：**部署图和QDQ量化图；动态batch，输入N×3×128×128、输出N×9 logits（实现推导）
- **可学习参数及更新：**原权重已训练；PTQ不做梯度训练
- **使用阶段与行为：**训练/验证/测试/推理：冻结后部署扩展；不用test选方案
- **采用原因与证据层：**ONNX是模型格式，ORT是运行时，PTQ是压缩方法
- **来源/函数/行号/版本：**extensions/quantization/quantization_extension.py::export_fp32 L173–187 @d5e4a71；extensions/quantization/quantization_extension.py::preprocess_onnx L190–195 @d5e4a71；extensions/quantization/quantization_extension.py::quantize_onnx L198–221 @d5e4a71

### P12b　保留早期5 Conv为FP32；左中

- **功能与输入：**18张训练诊断线索及固定节点排除清单；576张训练校准
- **输出与维度：**1个混合精度候选，保持指定5Conv浮点，验证F1.9956677882
- **可学习参数及更新：**不训练网络；MinMax校准获得尺度/零点，不是概率校准
- **使用阶段与行为：**训练/验证/测试/推理：只训练集诊断/校准、val工程门槛；无新外测
- **采用原因与证据层：**历史单次有限干预；不是唯一误差根因证明
- **来源/函数/行号/版本：**stage7/quant_candidate_plan.json；stage7/tools/ptq_candidate.py L1–末行 @d5e4a71；stage7/quantization_diagnosis.json

### P12c　第七阶段同次FP32与混合精度表；左下

- **功能与输入：**既有验证F1、byte与同次benchmark
- **输出与维度：**文件小69.9%；CPU中位.422792→.533642ms，慢26.2%
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：S7历史同次单线程batch1零张量；未新测速
- **采用原因与证据层：**文件大小、精度、速度是不同结果；不能以缩小推断加速
- **来源/函数/行号/版本：**stage7/evidence/ptq_candidate_early_float_verified/result.json；stage7/portable_demo/model/metadata.json

### P12d　独立FP32演示截图；右侧

- **功能与输入：**历史Mac断外网的内置val:0正确图像
- **输出与维度：**平滑肌预测及分列forward/含解码耗时
- **可学习参数及更新：**不学习；仅加载FP32模型
- **使用阶段与行为：**训练/验证/测试/推理：历史本机运行证据；本轮未启动模型/浏览器
- **采用原因与证据层：**无服务器/完整NPZ/Torch推理依赖；依赖安装仍需准备
- **来源/函数/行号/版本：**stage7/portable_demo/predict.py::Predictor L62–74 @d5e4a71；stage7/portable_demo/predict.py::preprocess L39–54 @d5e4a71；stage7/tests/browser_acceptance_minimal.json；portable_demo/samples/provenance.json

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P12-A1：冻结PyTorch权重 → FP32 ONNX|模型参数与计算图文件；冻结后的导出|得到可由ORT加载的图；**不表示：**不是换一个新的分类任务或重新训练|extensions/quantization/quantization_extension.py::export_fp32 L173–187 @d5e4a71|
|P12-A2：FP32 ONNX → 静态PTQ图|图与训练校准观察量导出的QDQ尺度/零点；576训练图像校准|构造8位与浮点转换计算图；**不表示：**不是使用val/test训练网络|extensions/quantization/quantization_extension.py::quantize_onnx L198–221 @d5e4a71|
|P12-A3：五具名Conv保留FP32 → 混合精度验证结果|固定配置执行后的模型文件与val指标；非直接数值流；先计划后校准和完整val|检验一次针对性干预；**不表示：**不是修好了ORT实际推理bug或证明每层INT8|stage7/quant_candidate_plan.json；ptq_candidate_early_float_verified/result.json|

## 第13页：结论与适用边界

### P13　三项结论、尚不能证明与工程交付；整页

- **功能与输入：**前述正式对照与工程记录
- **输出与维度：**有限结论＋未覆盖项；15训练、30正式CSV复核的历史事实
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：汇总历史结论，不是新实验
- **采用原因与证据层：**结论约束到当前任务/来源/协议；真实个人分工仍待填
- **来源/函数/行号/版本：**stage5/final/all_runs.csv；stage7/执行记录.md；stage7/尚未解决的问题.md

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第14页：谢谢，请老师批评指正（结束与索引）

### P14　结束与A1—A7索引；中下部

- **功能与输入：**七个附录超链接
- **输出与维度：**跳转15—21页；附录静态show=0
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：展示导航，无科学数据运算
- **采用原因与证据层：**页面/关系结构已读；未在PowerPoint真实点击验收
- **来源/函数/行号/版本：**reference/本轮实读PPT对象清单.json；03_当前PPT页序与对象清单.json

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第15页：为什么直接选择ResNet-18？

### P15　本轮要回答 / 没有回答 / 右侧G01重复图；左文字、右图

- **功能与输入：**第4页相同标准架构简图
- **输出与维度：**模型方案范围说明；机制模块和8条前向边复用P4-R/M条目
- **可学习参数及更新：**按第4页模块区分
- **使用阶段与行为：**训练/验证/测试/推理：模型选择范围解释，右图复用，非新实验
- **采用原因与证据层：**未比较ResNet34/50、从零训练或分别调参上限
- **来源/函数/行号/版本：**pathmnist_project/model.py::build_model L7–16 @d5e4a71；docs/方法与实验协议.md

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P15-R0：RGB图像64×64或128×128 → 卷积与池化|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P15-R1：卷积与池化 → 残差特征提取 / 快捷连接与主分支相加|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P15-R2：残差特征提取 / 快捷连接与主分支相加 → 全局平均池化|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P15-R3：全局平均池化 → 九类全连接层|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P15-M0：RGB图像64×64或128×128 → 卷积入口|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P15-M1：卷积入口 → 轻量倒残差特征提取 / 深度卷积、通道重标定|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P15-M2：轻量倒残差特征提取 / 深度卷积、通道重标定 → 卷积整合与全局平均池化|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|
|P15-M3：卷积整合与全局平均池化 → 九类分类器|首箭头为图像张量；中间为特征；进入末层为汇聚特征；同第4页前向机制，右侧重复图|串联特征提取与九类映射；**不表示：**不是权重文件、梯度、标签流，也不是残差捷径的单独画线|pathmnist_project/model.py::build_model L7–16 @d5e4a71；G01 source.json|

## 第16页：Batch size 128≠图像尺寸128×128

### P16　N×3×128×128及128→64→32；上部张量、下部试跑候选

- **功能与输入：**N批量、3颜色通道、H/W空间维；另列batch试跑顺序
- **输出与维度：**D完整batch N128；A/C空间64；最终batch可不足128
- **可学习参数及更新：**维度/超参数无可学习参数
- **使用阶段与行为：**训练/验证/测试/推理：正式训练128；CPU基准1；smoke16；PTQ校准16分开
- **采用原因与证据层：**按容量可行性从128起试，成功即返回，未证明64/32被实际跑过
- **来源/函数/行号/版本：**pathmnist_project/engine.py::loader L24–28 @d5e4a71；pathmnist_project/engine.py::probe L173–215 @d5e4a71；stage2/batch_probe.json

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P16-candidates：batch候选128 → 64再32|失败才降级的候选控制顺序；probe仅遇到OOM时下一候选|找共同可行batch；**不表示：**不表示三个候选都完成了精度/吞吐比较|pathmnist_project/engine.py::probe L173–215 @d5e4a71|

## 第17页：用癌相关间质的真实结果计算Macro-F1

### P17　TP/FP/FN、三公式和九类表；左计数公式、右九类F1

- **功能与输入：**已有D_seed2026混淆矩阵
- **输出与维度：**TP263 FP15 FN158；P.9460 R.6247 F1.7525；九类原始F1均值.929008
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：对已存混淆矩阵做简单算术；无概率重算
- **采用原因与证据层：**分类错误和患者漏诊不能混称；显示四舍五入
- **来源/函数/行号/版本：**pathmnist_project/metrics.py::classification_metrics L10–35 @d5e4a71；stage7/evidence/formal/D_seed2026/test_metrics.json；结果引用索引.csv

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第18页：E在测试集更高，为什么本轮仍用D？

### P18　G03选型—冻结—测试—描述分支；上部图

- **功能与输入：**历史val/CPU选型与冻结锁
- **输出与维度：**七条控制/文件/结果传递边；不反馈重选
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：测试前选择与测试后描述按时序分开
- **采用原因与证据层：**E测试更高可报告，不改写既定独立评价口径
- **来源/函数/行号/版本：**pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md

|箭头/关系|传递内容与发生时机|为何相连及不代表什么|来源|
|---|---|---|---|
|P18-A1：A—D验证组均值 → 容差候选集|验证指标/控制筛选；测试前|差最高≤.005；**不表示：**不是图像/梯度流|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md|
|P18-A2：容差候选集 → CPU前向最低配置D|CPU指标及选择控制；测试前|3次中位延迟再取中位数；**不表示：**不是报告中的算术均值|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md|
|P18-A3：配置D → D内验证最高seed|组内验证指标/排序控制；测试前|保留最优验证seed2026；**不表示：**不看test|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md|
|P18-A4：D内seed2026 → 冻结D_seed2026|权重/配置/哈希绑定；正式外测前|锁定所有预定轨迹来源；**不表示：**不冻结成不能如实报告E|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md|
|P18-A5：冻结D_seed2026 → 官方外部测试评价15条|控制许可与15份原权重；冻结后|统一评价预定矩阵；**不表示：**不只测试单个D、不训练test|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md|
|P18-A6：官方外部测试 → 保留全部正式结果|概率/指标/混淆矩阵文件；历史test完成|不择优删负面结果；**不表示：**不只保留最高分|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md|
|P18-A7：保留全部结果 → 测试后描述性分析|已保存CSV和指标；灰色虚线；历史测试查看后|描述D/E九类与置信度；**不表示：**不回流选模，不称预注册假设或新盲测|pathmnist_project/experiment.py::select_configuration L52–68 @d5e4a71；pathmnist_project/experiment.py::freeze L71–105 @d5e4a71；pathmnist_project/experiment.py::evaluate L125–146 @d5e4a71；stage7/analysis/definitions.md|

## 第19页：关闭增强后，并不是每个类别都改善

### P19　九类E−D横向条图；左图、右三类数值

- **功能与输入：**同seed E和D各类F1差
- **输出与维度：**横轴F1差；纵轴九类；均值±配对差样本SD；间质+.0467、平滑肌+.0327、碎屑−.0127
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：测试后描述；不选择新winner
- **采用原因与证据层：**保留全九类，方向与第10页D−E相反
- **来源/函数/行号/版本：**stage7/analysis/D_E_per_class_differences.csv；definitions.md；D_E_all_nine_classes.png

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第20页：量化完整口径：格式、精度与同次计时对照

### P20　ONNX / ORT / PTQ和三面板；顶部术语、下部图

- **功能与输入：**三次历史量化记录的val F1、byte、ms
- **输出与维度：**A:F1 0—1；B:MiB=bytes/2^20；C:CPU ms；菱形是各同次FP32基线
- **可学习参数及更新：**图表无；PTQ只估计量化范围，不做QAT梯度训练
- **使用阶段与行为：**训练/验证/测试/推理：S6-S/S6-U/S7分次；只val，无新官方test
- **采用原因与证据层：**保留原始失败和有限干预，不能混成统一四方案测速
- **来源/函数/行号/版本：**ppt_assets/quantization_values.csv及G04 source.json；两个comparison.json；S7 result.json

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 第21页：为什么没有开展患者/切片分组交叉验证？

### P21　数据可提供 / 实际采用 / K折限制；两列与底部句子

- **功能与输入：**图像、标签、数组索引；官方固定split与3seeds
- **输出与维度：**无可靠patient/slide分组；未实施K折，不将3seeds称3折
- **可学习参数及更新：**无
- **使用阶段与行为：**训练/验证/测试/推理：事实/局限说明；用户决定本轮不补K折和QAT
- **采用原因与证据层：**随机图像块折技术上可做，不能因此保证患者隔离；不虚构教师豁免
- **来源/函数/行号/版本：**stage1/manifest.json；reference/课程原文定位摘录.md；stage7/课程原文核实.md；本轮补充清单

本页没有需另释的科学流程箭头；图表连接线/导航若存在，不作参数或梯度传递解释。

## 写作时保持的证据层

1. 源码事实：项目调用、预处理、训练循环、冻结规则和推理接口；从版本化源码可定位。
2. 历史运行记录：模型效果、各次测速、真实/合成/断网/迁移路径验收；本轮只读保存记录，不重新声称设备验收。
3. 简单透明算术：保存组表的均值/样本SD、配对差、TP/FP/FN、bytes→MiB与相对百分比；见结果引用索引，不做30份逐图CSV完整指标重算。
4. 原理解释：标准架构简图中的作用说明，不当成该版第三方逐层源码核验、实测维度、机制因果或新增架构对照。

本文件用于ChatGPT进一步写最终图解，不替代最终长篇报告或讲稿。用户已决定不新增K折、QAT或Claude Code，课程对齐差异如实保留。
