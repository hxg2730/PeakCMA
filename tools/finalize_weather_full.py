"""Wait for the existing driver, then summarize and package its completed runs."""
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'Results/weather_full'


def main():
    while True:
        status = json.loads((ROOT / 'status.json').read_text(encoding='utf-8'))
        if status['status'] == 'failed':
            raise RuntimeError(status)
        if status['status'] == 'complete':
            break
        time.sleep(10)
    subprocess.run([sys.executable, 'tools/summarize_weather_full.py'], cwd=PROJECT, check=True)
    stats = json.loads((ROOT / 'full_statistics.json').read_text(encoding='utf-8'))
    report = json.loads((ROOT / 'final_report.json').read_text(encoding='utf-8'))
    lines = ['# Weather 全量训练记录', '',
             '实验已完成：完整 Weather 数据，输入 96 步、预测 192 步，六个随机种子的 TimeCMA 基线与 PeakCMA 配对实验。', '',
             '原始数据为 52,696 行、21 个动态变量。清洗后合法训练/验证/测试窗口为 33,010 / 5,078 / 10,349；未设置样本数上限。',
             '重复时间戳去重；缺失时刻补齐索引；-9999 转为缺测；仅补历史输入中不超过两步的内部缺口，未来目标不插值。标准化和峰值分位数只用训练集。', '',
             '使用真实冻结 GPT-2（FP16 推理），48,437 个窗口全部编码完成；缓存与预测网络使用 FP32。',
             '基线配置采用官方 Weather 192 步设置：channel=32、e_layer=1、d_layer=2、heads=8、batch=32。',
             'PeakCMA：omega=5、lambda_loc=0.1、lambda_peak=0.5；前四轮冻结骨干；新增层学习率 1e-4，骨干 1e-5。',
             '每次最多训练 100 轮，最少 50 轮、早停耐心值 50；最终权重按验证 MSE 选择，事件阈值按验证集选择。不会将早停后的轮数写成 100 轮。', '',
             '| 种子 | 基线实际轮数 | PeakCMA 实际轮数 | PeakCMA 最佳轮次 | 基线测试 MSE | PeakCMA 测试 MSE |',
             '|---|---:|---:|---:|---:|---:|']
    for row in stats['seed_rows']:
        seed = row['seed']
        lines.append(f"| {seed} | {row['baseline_epochs']} | {row['peak_epochs']} | "
                     f"{report['runs'][str(seed)]['best_epoch']} | {row['baseline_mse']:.6f} | {row['peak_mse']:.6f} |")
    lines += ['', '| 指标 | TimeCMA 基线（均值 ± 样本标准差） | PeakCMA（均值 ± 样本标准差） |',
              '|---|---:|---:|']
    for key in ('mse', 'mae', 'recall', 'precision', 'f1'):
        pair = stats['mean_and_sample_std'][key]
        values = [f"{pair[mode]['mean']:.6f} ± {pair[mode]['sample_std']:.6f}" for mode in ('baseline', 'peak')]
        lines.append(f'| {key} | {values[0]} | {values[1]} |')
    mse = stats['mean_and_sample_std']['mse']
    lines += ['', '结果解释：']
    if mse['peak']['mean'] > mse['baseline']['mean']:
        lines += ['本次全量实验中 PeakCMA 的平均整体 MSE 高于配对基线，未复现论文所述的整体性能优势。'
                  '峰值召回、误报与非峰值误差需要结合 full_statistics.json 一并判断，不能只凭召回率认定效果更好。']
    lines += ['事件按全部合法滑动预测窗口逐实例计数，重叠窗口中的事件重复计入；不应将计数直接视为去重的现实事件总数。',
              '这里的误差和事件指标来自当前清洗、提示词及评测实现；不能将其与论文表格作为完全相同协议的直接复现。',
              '实验范围仅覆盖 Weather 96→192，不包含所有数据集和所有预测长度。', '',
              '交付文件：六个 PeakCMA_Weather_full_seed*.pth；同种子基线 best_model.pth；逐轮 history.json、metrics.json、完整日志、数据窗口清单、配置、软件环境、配对统计及源码。',
              '2024 是预先固定的主交付种子，不按测试集挑选最佳种子。全部导出权重均已严格加载并与保存的测试预测核对。',
              '完整数据 CSV 包含在压缩包中；体积较大的 GPT-2 模型、嵌入缓存、逐窗口预测数组与断点续训状态保留在服务器原目录。',
              'artifact_manifest.json 逐文件记录 SHA-256；package_manifest.json 记录压缩包 SHA-256。', '',
              '复跑命令：', '```bash',
              'python tools/run_weather_full.py --device cuda --precision float16 --prompt_batch_size 64 --embedding_batch_size 4 --threads 8 --seeds 2024 42 1234 2025 2026 9999 --epochs 100 --preload_embeddings',
              '```', '']
    (ROOT / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    subprocess.run([sys.executable, 'tools/package_weather_full.py'], cwd=PROJECT, check=True)


if __name__ == '__main__':
    main()
