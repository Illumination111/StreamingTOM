"""StreamingTOM BioCache streaming evaluation adapter."""

from .adapter import StreamingTOMBioCacheAdapter, StreamingTOMConfig, derive_and_validate_num_layers
from .common import TimestampedFrame, iter_benchmark

__all__ = [
    "StreamingTOMBioCacheAdapter",
    "StreamingTOMConfig",
    "TimestampedFrame",
    "derive_and_validate_num_layers",
    "iter_benchmark",
]


def __getattr__(name):
    if name in {"QwenStreamAdapter", "StreamConfig"}:
        from .qwen_stream import QwenStreamAdapter, StreamConfig
        return {"QwenStreamAdapter": QwenStreamAdapter, "StreamConfig": StreamConfig}[name]
    raise AttributeError(name)


__all__ += ["QwenStreamAdapter", "StreamConfig"]
