"""Native Qwen CTR + OQM, reusing upstream selection and packed quantization."""
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
from streamingtom.modules.ctr import CTR
from streamingtom.modules.oqm import OQM
from streamingtom.modules.attention_processor import StreamingTOMAttentionProcessor
from .qwen_stream import clone_cache, pairs, cache_from, retrieval_cache, slice_cache


class Policy:
    name = 'StreamingTOM/Qwen2.5-VL'

    def __init__(self, config):
        self.config = config
        if config.group_size % (8 // config.quant_bits):
            raise ValueError('group_size must be aligned to the packed quantization width')
        self.reset()

    def release(self):
        handle = getattr(self, '_vision_handle', None)
        if handle is not None:
            handle.remove()
            self._vision_handle = None
        self._vision_scores = None

    def finish_query(self):
        self._vision_scores = None

    def reset(self):
        self.oqm = None
        self.ctr = None
        self.selector = None
        self.ctr_state = {}
        self.tail = None
        self.pending_features = None
        self.selection_trace = {}
        self._vision_scores = None

    def initialize(self, engine):
        self.ctr = CTR({'ctr_retain_tokens': self.config.retain_tokens,
                        'ctr_similarity_threshold': .8, 'ctr_k': 5, 'ctr_beta': .6})
        self.oqm = OQM({'oqm_retrieval_max_tokens': self.config.retrieval_tokens,
                        'oqm_enable_quantization': True, 'oqm_group_size': self.config.group_size,
                        'oqm_init_token_count': engine.system_cache.get_seq_length(),
                        'oqm_sliding_window_size': self.config.recent_tokens,
                        'oqm_quantization_bits': self.config.quant_bits})
        self.selector = StreamingTOMAttentionProcessor(None, self.oqm)
        for i, (k, v) in enumerate(pairs(engine.system_cache)):
            self.oqm.store_system_prompt('stream', i, k, v)
        # Remove the previous hook when re-opening a stream on the same model.
        old = getattr(self, '_vision_handle', None)
        if old is not None:
            old.remove()
        self._vision_handle = engine.model.model.visual.blocks[-1].attn.register_forward_pre_hook(
            self._observe_vision, with_kwargs=True)
        engine._hooks.append(self._vision_handle)

    def _observe_vision(self, module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        cu = kwargs.get('cu_seqlens', args[1] if len(args) > 1 else None)
        q, k, _ = module.qkv(hidden).reshape(hidden.shape[0], 3, module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, *kwargs['position_embeddings'])
        scores = []
        # Observe incoming attention without replacing the vision forward result.
        for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
            query, key = q[start:end].transpose(0, 1), k[start:end].transpose(0, 1)
            accumulated = torch.zeros(end-start, device=hidden.device, dtype=torch.float32)
            for offset in range(0, end-start, 128):
                attn = (query[:, offset:offset+128].float() @ key.float().transpose(-1, -2)) * module.scaling
                accumulated += attn.softmax(-1).sum((0, 1))
            scores.append(accumulated / ((end-start) * module.num_heads))
        self._vision_scores = torch.cat(scores)

    def ingest(self, engine, features, positions, grid):
        if self._vision_scores is None:
            raise RuntimeError('native vision attention was not observed')
        visual = engine.model.model.visual
        merge = visual.spatial_merge_size ** 2
        window_index, _ = visual.get_window_index(grid)
        attention = self._vision_scores.reshape(-1, merge).mean(-1)[window_index.argsort()].unsqueeze(0)
        self._vision_scores = None
        self.ctr.retain_tokens = min(self.config.retain_tokens, features.shape[1])
        static, dynamic = self.ctr._classify_tokens(features, self.ctr_state)
        compressed = self.ctr._compress_single_frame(0, features, attention, static, dynamic).unsqueeze(0)
        pos = positions[:, :, self.ctr.last_selected_positions]
        if self.pending_features is not None:
            compressed = torch.cat((self.pending_features[0], compressed), 1)
            pos = torch.cat((self.pending_features[1], pos), 2)
        count = compressed.shape[1] // self.config.group_size * self.config.group_size
        self.pending_features = ((compressed[:, count:].clone(), pos[:, :, count:].clone())
                                 if compressed.shape[1] > count else None)
        if not count:
            return
        cache = self._window_cache(engine)
        output = engine.forward(compressed[:, :count], cache, pos[:, :, :count])
        new = [(k[:, :, -count:].clone(), v[:, :, -count:].clone()) for k, v in pairs(output.past_key_values)]
        for i, (k, v) in enumerate(new):
            self.oqm.store_kv_cache('stream', i, k, v)
            keys = k[0].transpose(0, 1).flatten(1)
            self.oqm.store_token_keys_as_groups('stream', i, keys)

    def _window_cache(self, engine):
        # Official OQM window reconstruction includes the exact system prefix.
        return cache_from([self.oqm.get_windowed_kv('stream', i)
                           for i in range(len(engine.system_cache.layers))])

    def tail_context(self, engine):
        return self._window_cache(engine)

    def query_cache(self, engine, embeddings):
        self.selection_trace = {}
        def select(i, query):
            k, v = pairs(engine.system_cache)[i]
            keys = self.oqm.get_group_keys('stream', i)
            if keys is None or not keys.shape[0]:
                return k.clone(), v.clone()
            groups = query.shape[1] // k.shape[1]
            q = query.reshape(1, k.shape[1], groups, query.shape[2], query.shape[3]).mean(2)
            indices = self.selector._select_top_k_groups(q[0].mean(1).flatten(), keys,
                                                        self.config.retrieval_tokens)
            self.selection_trace[i] = indices.cpu().tolist()
            return self.oqm.get_selective_kv('stream', i, indices)
        cache = retrieval_cache(engine.system_cache, select)
        if self.pending_features is not None:
            features, positions = self.pending_features
            output = engine.forward(features, self._window_cache(engine), positions)
            cache.stream_extra_tail = slice_cache(output.past_key_values, -features.shape[1])
        return cache

    def state(self):
        if self.oqm is None:
            return {}
        return {'quantized_archive': self.oqm.quantized_storage, 'group_keys': self.oqm.group_keys,
                'batch_metadata': self.oqm.batch_metadata, 'stored_tokens': self.oqm.stored_tokens_count,
                'ctr_state': self.ctr_state, 'last_centroid_indices': getattr(self.ctr, 'last_selected_positions', None), 'pending_features': self.pending_features,
                'retrieval_trace': self.selection_trace, 'vision_scores': self._vision_scores}
