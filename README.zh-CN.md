# CuraView 推理框架

CuraView 将出院小结拆成句子，从同一患者的记录中检索证据，再判断证据能否支持每句话。[英文说明](README.md) · [论文](https://arxiv.org/abs/2605.03476)

## 怎样运行推理

1. 使用 Python 3.12，按[构图说明](graph/README.md)为有权处理的记录建立 GraphRAG 2.7 索引。建图和检测使用同一标识，例如 `patient_1`。
2. 安装[检测依赖](detection/requirements.txt)，在[配置文件](detection/config.yaml)中指定索引和模型。实际推理需要可用的模型服务或兼容的本地模型。输入、日志和模型响应保存在获准使用的环境中。
3. 执行 `python detection/detect.py /path/to/summary.txt --patient-id patient_1`。[检测说明](detection/README.md)列出了证据等级、结构化输出字段，以及检索或判断失败时的处理方式。

| 从论文核对哪一步 | 代码与说明 |
|---|---|
| 患者记录如何建图、取回证据 | [构图、查询与输入格式](graph/README.md) |
| 每句话如何判定 | [检测提示词](detection/doc/prompts/)、[输出校验](detection/curaview/_03_detection/parsing/output_contract.py)、[离线检测示例](detection/smoke_test.py) |
| 同一检测器怎样比较三种证据检索方式 | [Flat、Generic、Domain 三组比较](comparison/README.md)与[本地模拟服务集成测试](graph/test_native.py)，这项比较单独使用 GraphRAG 3.3 环境。 |
| 如何计算预测结果的指标 | [评分程序与四句示例](evaluation/README.md) |
| 如何在本机查看结果 | [Web 界面](webui/README.md) |

在仓库根目录运行 `python evaluation/synthetic_check.py`，只用 Python 标准库即可核对四句虚构输入的精确率、召回率与 F1。安装检测依赖后，运行 `python detection/smoke_test.py` 可以用模拟模型检查检索返回的证据如何进入判定流程。

论文说明验证模型的训练方法与超参数。本次公开推理代码、设置、提示词和虚构测试；训练与训练数据整理代码、训练日志、检查点及研究数据均在公开范围之外。复现论文的临床性能表还需要相应历史输入与模型。

引用与许可证见[英文首页](README.md#citation)及 [MIT 许可证](LICENSE)。
