"""StreamingTOM/OQM 的 BioCache 在线流适配器。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .common import BUDGET_TIERS, PhaseTimer, TimestampedFrame, describe_tree, object_nbytes


@dataclass
class StreamingTOMConfig:
    segment_frames: int = 32
    budget_tier: str = "25%"
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.segment_frames < 1:
            raise ValueError("segment_frames must be positive")
        if self.budget_tier not in BUDGET_TIERS:
            raise ValueError(f"budget_tier must be one of {sorted(BUDGET_TIERS)}")


def derive_and_validate_num_layers(model: Any) -> int:
    """从 config 推导层数，并与模型结构显式校验；绝不默认 28。"""
    config = getattr(model, "config", None)
    configured = (
        getattr(getattr(config, "text_config", None), "num_hidden_layers", None)
        or getattr(config, "num_hidden_layers", None)
    )
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    if configured is None:
        raise ValueError("cannot derive num_hidden_layers from model config")
    if layers is None:
        raise ValueError("cannot locate model language layers")
    if int(configured) != len(layers):
        raise ValueError(f"configured layers ({configured}) != actual layers ({len(layers)})")
    return int(configured)


class _OQMWorkspaceProbe:
    """记录 OQM 反量化路径实际返回张量的峰值字节，不持有张量引用。"""

    METHODS = ("_unpack_Nbit", "_dequantize_tensor", "_reconstruct_selective_kv", "_reconstruct_windowed_kv")

    def __init__(self) -> None:
        self.peak_bytes = 0
        self.components: dict[str, int] = {}
        self._originals: dict[str, Any] = {}
        self._oqm: Any = None

    def install(self, oqm: Any) -> None:
        if self._oqm is oqm:
            return
        self.restore()
        self._oqm = oqm
        for name in self.METHODS:
            original = getattr(oqm, name)
            self._originals[name] = original

            def wrapped(*args: Any, __name: str = name, __original: Any = original, **kwargs: Any):
                result = __original(*args, **kwargs)
                size = object_nbytes(result)
                self.components[__name] = max(self.components.get(__name, 0), size)
                self.peak_bytes = max(self.peak_bytes, size)
                return result

            setattr(oqm, name, wrapped)

    def restore(self) -> None:
        if self._oqm is not None:
            for name, original in self._originals.items():
                setattr(self._oqm, name, original)
        self._originals.clear()
        self._oqm = None

    def report(self) -> dict[str, Any]:
        return {
            "peak_returned_tensor_bytes": self.peak_bytes,
            "components": dict(self.components),
            "workspace_inventory": [
                "selected packed K/V slices",
                "unpacked uint8 K/V",
                "scale/min broadcast operands",
                "dequantized group K/V",
                "reconstructed output K/V",
            ],
        }


class StreamingTOMBioCacheAdapter:
    def __init__(self, model: Any = None, config: Optional[StreamingTOMConfig] = None) -> None:
        self.model = model
        self.config = config or StreamingTOMConfig()
        self.timer = PhaseTimer()
        self.stream_id: Optional[str] = None
        self._pending: list[TimestampedFrame] = []
        self._timestamps: list[float] = []
        self._frames_pushed = self._segments_pushed = self._answers = 0
        self._started_backend = False
        self._opened_at: Optional[float] = None
        self._num_layers: Optional[int] = None
        self._workspace = _OQMWorkspaceProbe()
        self._temporary_peaks: dict[str, int] = {}

    def open_stream(self, stream_id: Optional[str] = None) -> str:
        if self.stream_id is not None:
            raise RuntimeError("stream already open")
        if self.model is not None and not self.config.dry_run:
            self._num_layers = derive_and_validate_num_layers(self.model)
            if not callable(getattr(self.model, "generate_with_streamingtom_streaming", None)):
                raise ValueError("model is not patched with generate_with_streamingtom_streaming")
        self.stream_id = stream_id or f"streamingtom-{uuid.uuid4().hex[:12]}"
        self._opened_at = time.time()
        self.reset()
        return self.stream_id

    def reset(self) -> None:
        self._pending.clear()
        self._timestamps.clear()
        self._frames_pushed = self._segments_pushed = self._answers = 0
        self._started_backend = False
        self.timer = PhaseTimer()
        self._temporary_peaks.clear()
        self._workspace.restore()
        if self.model is not None and not self.config.dry_run:
            pipelines = getattr(self.model, "_video_pipelines", {})
            with self.timer.measure("reset"):
                for video_id, pipeline in list(pipelines.items()):
                    pipeline.streamingtom_core.clear_cache(video_id)
                pipelines.clear()

    def push_frames(self, frames: Iterable[TimestampedFrame | tuple[float, Any]]) -> dict[str, Any]:
        """只消费外部提供帧，至多缓存当前 segment；不接受路径且不读取未来帧。"""
        self._require_open()
        accepted = 0
        for item in frames:
            with self.timer.measure("ingest"):
                stamped = item if isinstance(item, TimestampedFrame) else TimestampedFrame(float(item[0]), item[1])
                if self._timestamps and stamped.timestamp < self._timestamps[-1]:
                    raise ValueError("frame timestamps must be monotonic")
                self._pending.append(stamped)
                self._timestamps.append(stamped.timestamp)
                self._frames_pushed += 1
                accepted += 1
            if len(self._pending) >= self.config.segment_frames:
                self._flush_segment()
        return {"accepted_frames": accepted, "total_frames": self._frames_pushed,
                "pending_frames": len(self._pending)}

    def _flush_segment(self) -> None:
        if not self._pending:
            return
        segment, self._pending = self._pending, []
        if not self.config.dry_run:
            if self.model is None:
                raise RuntimeError("a StreamingTOM model is required outside dry-run")
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("torch is required to encode StreamingTOM frames") from exc
            payloads = [item.frame for item in segment]
            if not all(isinstance(frame, torch.Tensor) for frame in payloads):
                raise TypeError("StreamingTOM expects externally preprocessed torch.Tensor frames [C,H,W]")
            images = torch.stack(payloads)
            self._record_temporary("encoded_segment", images)
            with self.timer.measure("encode"):
                self.model.generate_with_streamingtom_streaming(
                    images=images, questions=[], video_id=self.stream_id,
                    is_start=not self._started_backend,
                )
            self._started_backend = True
            oqm = self._get_oqm()
            if oqm is not None:
                self._workspace.install(oqm)
        self._segments_pushed += 1

    def answer(self, question: Any, **kwargs: Any) -> Any:
        self._require_open()
        if self._pending:
            self._flush_segment()
        if self.config.dry_run:
            result: Any = {"dry_run": True, "question": question}
        else:
            tokenizer = getattr(self.model, "tokenizer", None) if self.model is not None else None
            if isinstance(question, str):
                if tokenizer is None:
                    raise ValueError("string questions require model.tokenizer")
                tokenized = tokenizer(question, return_tensors="pt")
                input_ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized["input_ids"]
            else:
                input_ids = question
            self._record_temporary("query_input", input_ids)
            with self.timer.measure("query"):
                answers = self.model.generate_with_streamingtom_streaming(
                    images=None, questions=[{"batch_idx": -1, "input_ids": input_ids}],
                    video_id=self.stream_id, is_start=False, **kwargs,
                )
            result = answers[0]["answer"] if answers else None
            self._record_temporary("query_output", result)
        self._answers += 1
        return result

    query = answer

    def _get_oqm(self) -> Any:
        if self.model is None or self.stream_id is None:
            return None
        pipeline = getattr(self.model, "_video_pipelines", {}).get(self.stream_id)
        return getattr(getattr(pipeline, "streamingtom_core", None), "oqm", None)

    def persistent_state(self) -> dict[str, Any]:
        """返回 OQM 随流累计、需纳入长期内存核算的状态对象。"""
        oqm = self._get_oqm()
        if oqm is None:
            return {"external_timestamps": self._timestamps}
        return {
            "packed_tensors_scale_min_and_lengths": getattr(oqm, "quantized_storage", {}),
            "unquantized_kv": getattr(oqm, "kv_cache_storage", {}),
            "group_keys": getattr(oqm, "group_keys", {}),
            "batch_metadata": getattr(oqm, "batch_metadata", {}),
            "stored_tokens_count": getattr(oqm, "stored_tokens_count", {}),
            "original_dtype": getattr(oqm, "original_dtype", {}),
            "external_timestamps": self._timestamps,
        }

    def static_extra_state(self) -> dict[str, Any]:
        """返回模型、tokenizer、OQM 配置和适配器配置等静态状态。"""
        oqm = self._get_oqm()
        oqm_config = {} if oqm is None else {
            key: getattr(oqm, key, None) for key in (
                "retrieval_max_tokens", "enable_quantization", "group_size",
                "init_token_count", "sliding_window_size", "quantization_bits",
                "quantization_levels", "pack_size",
            )
        }
        parameters = list(self.model.parameters()) if self.model is not None and hasattr(self.model, "parameters") else []
        return {"model_parameters": parameters, "model_config": getattr(self.model, "config", None),
                "tokenizer": getattr(self.model, "tokenizer", None),
                "oqm_config": oqm_config, "adapter_config": self.config}

    def temporary_state(self) -> dict[str, Any]:
        """返回当前段缓存、临时峰值和反量化工作区记录。"""
        return {
            "pending_external_frames": self._pending,
            "observed_peak_bytes": self._temporary_peaks,
            "dequantization_workspace": self._workspace.report(),
        }

    def state_report(self) -> dict[str, Any]:
        persistent = self.persistent_state()
        static = self.static_extra_state()
        temporary = self.temporary_state()
        oqm = self._get_oqm()
        return {
            "persistent": {"bytes": object_nbytes(persistent), "objects": describe_tree(persistent, "persistent")},
            "static": {"bytes": object_nbytes(static), "objects": describe_tree(static, "static")},
            "temporary": {"bytes": object_nbytes(temporary), "objects": describe_tree(temporary, "temporary"),
                          "dequantization": self._workspace.report()},
            "budget": {"tier": self.config.budget_tier,
                       "ratio": BUDGET_TIERS[self.config.budget_tier]["ratio"],
                       "retrieval_max_tokens": getattr(oqm, "retrieval_max_tokens", None),
                       "sliding_window_size": getattr(oqm, "sliding_window_size", None)},
        }

    def report(self) -> dict[str, Any]:
        return {"stream_id": self.stream_id, "opened_at": self._opened_at,
                "frames": self._frames_pushed, "segments": self._segments_pushed,
                "answers": self._answers, "num_layers": self._num_layers,
                "timing": self.timer.report(), "state": self.state_report()}

    def close_stream(self) -> dict[str, Any]:
        self._require_open()
        if self._pending:
            self._flush_segment()
        report = self.report()
        self._workspace.restore()
        if self.model is not None and not self.config.dry_run:
            pipeline = getattr(self.model, "_video_pipelines", {}).pop(self.stream_id, None)
            if pipeline is not None:
                with self.timer.measure("close"):
                    pipeline.streamingtom_core.clear_cache(self.stream_id)
        report["timing"] = self.timer.report()
        self.stream_id = None
        return report

    def _require_open(self) -> None:
        if self.stream_id is None:
            raise RuntimeError("call open_stream() first")

    def _record_temporary(self, name: str, value: Any) -> None:
        self._temporary_peaks[name] = max(
            self._temporary_peaks.get(name, 0), object_nbytes(value)
        )
