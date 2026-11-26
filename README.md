# PeakCMA

按提供的 PeakCMA 论文实现的研究复现代码。主模型为 `models/PeakCMA.py`，原始交叉模态骨干保留在 `models/backbone.py`，用于同种子基线预训练和严格加载权重。原项目的许可与作者权利保留在 LICENSE.txt；基线来源为 TimeCMA (AAAI 2025)。

## 已实现

- 公式 (7)-(8)：预测步嵌入、64 单元 GELU 定位特征、sigmoid 分数、64 单元有符号残差头；残差末层零初始化，只对指定变量加修正。
- 公式 (9)：全变量 MSE、加权 soft-label BCE、峰值加权 MAE，默认 omega=5、lambda_loc=0.1、lambda_peak=0.5。
- 训练集 95% 分位数、局部极值、平台取最早、确定性 NMS、sigma=1 且截断到 3 sigma 的软标签。
- 前四轮冻结整个骨干；新增模块学习率 1e-4，骨干 1e-5；AdamW、cosine、梯度裁剪 5、验证集 MSE 早停。前向传播不接收未来标签。
- 验证集选择定位阈值并固定用于测试；一对一事件匹配先最大化匹配数，再最小化时间距离。输出 TP/FP/FN、召回率、精确率、幅度偏差、幅度/时间/邻域/非峰值误差。

## 安装与数据

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

CSV 放入 `dataset/Weather.csv`、`dataset/Weather2K-R.csv` 或 `dataset/ERA5.csv`，含 `date` 和动态变量。多站点/格点使用 `location` 列区分时间序列；latitude/longitude/altitude 不作为目标。ERA5 自动从 u10/v10 派生 WS10，需要通过 `--columns` 明确指定论文使用的十个动态变量（包含 WS10）。本项目没有原始 NetCDF/Weather2K 二进制数据转换器，需先整理为该 CSV 格式。

Weather 默认峰值列名 wv,rain；标准公开文件若为 `wv (m/s)`、`rain (mm)`，命令均附加 `--peak_columns "wv (m/s),rain (mm)"`。Weather2K-R 默认 Tair,WS；ERA5 默认 t2m,WS10。通过 `--columns` 设置完整动态变量的顺序，基线和微调必须一致。

Weather 使用论文固定边界 36887/42156/52696（完整数据）；非完整 Weather CSV 使用 7:1:2，属于调试配置。Weather2K-R 使用论文日期边界；ERA5 每个格点按 7:1:2。所有目标窗口必须在一个划分内，历史允许使用之前划分的观测。只补历史内部不超过两步的缺失，目标不补值。Weather2K-R 标准化统计在训练站点池化，ERA5 按格点；极值分位数均按变量池化训练观测。

## 运行

从项目根目录运行，三个步骤使用相同数据和参数：

```powershell
.venv\Scripts\python.exe -m storage.store_emb --data_path Weather --num_workers 0
.venv\Scripts\python.exe train.py --data_path Weather --mode baseline --seed 2024 --num_workers 0
.venv\Scripts\python.exe train.py --data_path Weather --mode peak --seed 2024 --num_workers 0 --baseline_checkpoint Results/PeakCMA/Weather/96_192/2024/baseline/best_model.pth
```

首次生成缓存需下载 GPT-2。冻结 GPT-2 使用 eval 模式，Prompt 5 保留三位小数，逐变量缓存最后 token。新缓存按数据内容哈希、输入/输出长度和划分隔离，旧项目的整数化 prompt 缓存不能直接复用。缓存键也包含列顺序和提示词版本。

Linux/Git Bash 下 `bash scripts/Weather.sh`、`Weather2K-R.sh`、`ERA5.sh` 顺序执行六组种子 42/1234/2024/2025/2026/9999 的基线和微调。脚本固定论文峰值实验 horizon；若改长度，请使用上面的独立命令。

结果：`Results/PeakCMA/<dataset>/<input>_<output>/<seed>/<mode>/`，含最佳权重、metrics.json、test_predictions.npz。baseline 模式仅用于提供论文要求的配对基线。

## 复现边界

论文没有明确所有数据的骨干宽度/编码解码层数、训练总轮数、定位阈值搜索网格及 ERA5 十变量清单。这里沿用骨干默认宽度 32、编码/解码各一层，总轮数 100；阈值在 [0,1] 以 0.01 搜索，F1 并列取较小值。须以作者实验配置进一步核准，不能据此承诺论文数值一致。骨干保留原实现中的 cross-modal 残差/FFN结构以兼容现有 checkpoint。

metrics.json 中的峰值误差是绝对指标；论文 baseline=100 的相对指标需配对运行后计算：同一变量 PeakCMA 误差 / baseline 误差 × 100，再对有效变量宏平均。无匹配/无事件的未定义指标输出 null。本实现当前提供高尾事件（论文实验全部为高尾），不自动把其他变量视为峰值变量。训练尚需真实数据、GPT-2 缓存及完整六种子实验验证；本仓库不包含或伪造论文结果。

`python -m unittest discover -s tests -v` 执行结构、事件协议与数据边界测试。

工作区外层目录仍为 TimeCMA-main；内部项目和运行产物均已切换为 PeakCMA。可以关闭工作区后在资源管理器将外层目录改名为 PeakCMA-main。

## 真实 Weather 最小训练

已提供 `tools/run_weather_minimal.py`，从标准 Weather 全量 CSV 的原始划分中均匀选取 32/8/8 个真实窗口，保留全部 21 个变量和 96/192 输入输出长度。标准 CSV 存在相同重复记录及缺失时刻：去除相同重复记录，补齐时间索引但不补长缺口，划分边界保留原始索引对应的时间戳。训练集统计仍使用完整训练区间；窗口限制只控制计算量。

准备好 `dataset/Weather.csv` 和 `models/pretrained/gpt2/` 后运行：

```powershell
.venv\Scripts\python.exe tools/run_weather_minimal.py
```

脚本离线加载官方 GPT-2 Base，生成真实三位小数 Prompt 5 嵌入，训练 10 轮基线、8 轮 PeakCMA。结果和日志保存在 `Results/weather_real/`；最终 `PeakCMA_Weather_minimal.pth` 包含模型权重、列顺序、标准化统计、训练参数及事件阈值。重新运行会复用已有嵌入。它是小样本真实链路验证，不代表完整数据、六种子论文实验。

## 缺测修正后的增强训练

`python tools/run_weather_corrected.py` 使用新版 `sentinel-clean-v2`：将动态变量中的 -9999 缺测编码转为 NaN，训练统计忽略缺测，目标包含缺测的窗口剔除，历史仍只插补内部不超过两步的缺口。原始 CSV 保持不变。缓存键包含预处理版本，基线加载会验证预处理版本与标准化统计，防止旧权重混用。

此次 CPU 实验保留先前合法窗口，再按时间均匀补充到训练/验证/测试 64/16/32 个窗口，不按测试峰值标签挑选数据。仅对历史数值与时间未改变的 Prompt 5 复用官方 GPT-2 缓存；无效或被插补的历史重新计算。

基线最多 80 轮，PeakCMA 最多 100 轮（至少 50 轮后才允许早停），每轮 8 次更新。骨干前 4 轮冻结，骨干的 cosine 学习率从解冻时才开始衰减。为了让小样本实验关注定位，微调启用 `--selection_metric joint`，按验证集 MSE + 0.1 * 定位 BCE + 0.5 * 峰值 MAE 选权重；默认命令仍按论文使用 MSE。此项属于明确记录的实验配置差异。定位标签、三项训练损失与门控结构保持原定义，不使用测试集调参。

`history.json` 保存分项训练/验证损失及各组学习率。新权重和全部日志保存到 `Results/weather_corrected/`，旧实验保留作审计。不同预处理下的总体 MSE 不能直接相减比较；新 PeakCMA 应与同次清洗、同窗口的 baseline 比较。可运行 `python tools/analyze_peak_run.py --run_dir Results/weather_corrected` 诊断新实验。

### 定位优先的受约束对照

第一轮清洗后重训采用联合损失选权重，但验证集定位能力仍弱，因此又运行了同缓存、同窗口、同基线的对照实验：新增模块学习率 1e-3、lambda_loc=0.5、lambda_peak=0.1，最多 100 轮，采用 `--selection_metric event_f1 --max_validation_mse 0.633342`。MSE 上限来自该次基线最佳验证 MSE 的约 1.02 倍，未用测试结果设置。权重和阈值均按验证集选择；阈值 F1 并列时先取更少误报，再取较高阈值。

`Results/weather_balanced/` 保存这轮输出，训练日志位于 `Results/weather_corrected/balanced_training.log`。执行 `python tools/finalize_balanced_run.py` 可以重新验证加载输出、检查配对窗口并导出带阈值的权重和报告。参数会完整写入 `training_config.json`。这些训练系数和选权重策略是实验配置差异，不应当作论文默认配置；模型公式未修改。

## Weather 全量 GPU 实验

原始 CSV 完整包含 52,696 行、21 个动态变量；全量模式不设样本上限，按缺测处理规则获得 train/val/test = 33,010/5,078/10,349 个窗口。运行命令：

```bash
python tools/run_weather_full.py --device cuda --precision float16 --prompt_batch_size 64 --embedding_batch_size 4 --threads 8 --seeds 2024 42 1234 2025 2026 9999 --epochs 100
```

该命令执行 Weather 96→192 的六组配对实验，使用官方 TimeCMA Weather 192 步配置（channel=32、e_layer=1、d_layer=2、batch=32、dropout=0.1、基线学习率 1e-4）。PeakCMA 使用论文损失系数 0.1/0.5/omega=5、前四轮冻结、骨干学习率 1e-5，按验证 MSE 选权重；不使用小样本实验的定位优先系数。GPT-2 使用 FP16 编码加速，缓存和预测模型为 FP32；精度与软件版本写入 configuration.json。该实验范围是 Weather 的 192 步峰值设置，不等于所有数据集和所有预测长度的完整论文实验。

缓存写入和训练检查点均采用临时文件原子替换。已有缓存会校验形状与有限值，损坏文件重新计算。训练通过 `--resume auto` 恢复模型、优化器、调度器、早停状态及 CPU/CUDA/NumPy/Python 随机状态；同配置重新执行全量命令即可续跑。恢复时验证配置及数据指纹，不能混用不同处理版本。

日志、数据窗口索引、环境版本、逐轮损失、状态、最终每种子权重保存在 Results/weather_full。`python tools/full_status.py` 查看进度；六组完成后 `python tools/summarize_weather_full.py` 计算 n-1 样本标准差、配对置信区间、双侧配对检验、峰值相对指标及池化事件计数。`--partial` 仅输出已完成配对的临时统计，不能当作六种子的最终结果。
