# StreamingTOM BioCache streaming adapter

该包包装官方 `generate_with_streamingtom_streaming` 与 OQM。调用方必须按时间顺序提供已经预处理的 `[C,H,W]` 外部 tensor；适配器不接收视频路径、不解码、不下载，也不会预读完整视频。

## 契约

- `open_stream` / `reset`：验证模型层数并创建或清理流。
- `push_frames`：最多持有一个 `segment_frames` 段，随后调用官方 streaming 入口；段内官方 encoder batch 行为保持不变。
- `answer` / `query`：对字符串用模型 tokenizer 编码，或直接接收 `input_ids`，再调用官方 query/OQM 路径。
- `close_stream`：返回报告并释放该 video id。
- `persistent_state()` / `static_extra_state()` / `temporary_state()`：公开返回统一契约的三类原始状态对象。
- `state_report()`：复用上述三个方法，递归按 tensor storage 计算 `persistent`、`static`、`temporary` 的真实字节。

persistent 清单包含未量化 KV，以及每层 OQM `init_tokens`、`keys/values_packed`、`keys/values_scales`、`keys/values_mins`、`keys/values_S`、group keys、batch metadata、stored token count 与 original dtype。temporary 清单包含外部段缓存和反量化工作区；probe 会记录 unpack、dequantize、selective/window reconstruction 实际返回 tensor 的峰值，并明确列出 packed slice、uint8 unpack、scale/min broadcast、dequantized groups、输出 K/V。

层数从 `config.text_config.num_hidden_layers` 或 `config.num_hidden_layers` 推导，并与实际 `layers` 长度显式校验。官方路径中的 28 层硬编码也已替换为同一校验，任何不一致都会报错。

预算档 `12.5%`、`25%`、`50%` 作为统一报告标签；报告同时保留 OQM 的实际 `retrieval_max_tokens` 和 `sliding_window_size`，适配器不会静默改变官方配置。计时分 ingest、encode、query、reset、close。

## Dry-run

```bash
python -m biocache_adapt.cli --dry-run --budget-tier 25%
python -m biocache_adapt.cli --dry-run --benchmark StreamingBench \
  --annotations /local/path/streamingbench.jsonl
```

支持 OVO、StreamingBench、TwinPast、VNBench、EgoLifeQA 本地 JSON/JSONL 的统一 schema，不下载数据。运行产出 `manifest.json`、`metrics.jsonl` 和可选 schema preview。PyTorch 与 StreamingTOM 均延迟到真实编码时使用。
