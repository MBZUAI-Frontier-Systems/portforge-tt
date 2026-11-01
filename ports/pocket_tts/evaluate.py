"""Trusted evaluator for the Kyutai Pocket TTS (CALM) port to Tenstorrent.

The shared scaffold is portforge_eval's evaluator kit; this file holds what is
specific to Pocket TTS: the suites, the gated checkpoint, the official
implementation as the oracle, the backend's generate()/predict_latent() calls
and the autoregressive gates.

Public inputs per case: prepared text token ids, the frozen noise pool rows the
sampler must consume (one row per generated frame, in order), per-voice
precomputed ModelState KV caches from the released voice-embedding files, and
bounded teacher-forced anchor prefixes (oracle latents[:t]) on development
stages only. Never exposed: full oracle latents, oracle audio, oracle EOS.

The model is autoregressive over continuous 32-dim Mimi latents at 12.5 Hz with
one-step LSD flow decoding; there are no discrete tokens, so numerical gates
compare bounded teacher-forced next-frame predictions (no drift) and free-run
sequences agree on the EOS frame within a tolerance, with the WER gate on the
full stage as the end-to-end task metric. Text preparation (chunking,
normalization, tokenization) and voice-prompt state loading are deterministic
preprocessing on the host CPU; learned compute must run on TT.

Reference policy notes (declared, reviewable):
  * The official implementation draws per-frame sampler noise from the global
    torch RNG (CUDA RNG on a GPU oracle). A narrow patch of
    torch.nn.init.normal_ feeds the model the frozen public pool instead, so
    the FP32 oracle and the TT candidate consume byte-identical noise rows in
    frame order. Nothing else is intercepted.
  * Oracle audio comes from the official TTSModel.generate_audio. Oracle
    latents/EOS come from a faithful replay of the official autoregressive
    loop (init_states, _expand_kv_cache, _run_flow_lm_and_increment_step) with
    the same pool; both passes must consume the same number of rows.
"""
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import portforge_eval
from portforge_eval import (Progress, check_suite, reference_execution, suite_hash,  # noqa: F401
                            validate_input_preservation, validate_receipt, write_json)
from portforge_models import model_spec, verify_checkpoint

MODEL = "kyutai/pocket-tts"
REVISION = "7f364c05d1be3faf2b234cdbe4cbb68a026d4218"
WEIGHT_SHA256 = "7767ce79b1b8db523d867e70855a00627aacfe44aaa5c1fd7ebe135fba71fff9"
CORPUS_SHA256 = "2160008909385c5384b2e3fc7fa72ada9b463aa5c21e189e5189b5a15a187964"
GATES = {"nrmse": .04, "free_run_nrmse": .10, "eos_frame_tolerance": 2,
         "behavior_max_nrmse": .01, "audio_mel_nrmse": .12, "max_wer_inflation": .02}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "tokens_voice_noise_to_audio_with_sync_excluding_load_v1"
KEY = "pocket_tts"

SAMPLE_RATE = 24000            # Mimi output audio rate
FRAME_RATE = 12.5              # generated latent frames per second
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)   # 1920
LATENT_DIM = 32
TRANSFORMER_LAYERS = 6
TEMPERATURE = 0.3
NOISE_STD = TEMPERATURE ** 0.5
DECODE_STEPS = 1
EOS_THRESHOLD = -4.0
MAX_TEXT_TOKENS = 50
MIN_FRAMES_BEFORE_EOS = 6
TOKENS_PER_SECOND_ESTIMATE = 3.0
GEN_SECONDS_PADDING = 2.0
FRAMES_AFTER_EOS_EXTRA = 2
ANCHOR_FRAMES = (4, 12, 32)    # development stages only; never exported at full
VOICES = ("alba", "vera", "michael")
ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
ASR_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"

# Flat checkpoint layout: pinned HF subpaths at the pinned revision. The gated
# repository stores the recipe in subdirectories this registry cannot express;
# prepare_checkpoint() downloads exactly these bytes and flattens them.
HF_PATHS = {
    "model.safetensors": "languages/english_2026-09_2/model.safetensors",
    "tokenizer.json": "languages/english_2026-09_2/tokenizer.json",
    "tokenizer.model": "languages/english_2026-09_2/tokenizer.model",
    "voice_alba.safetensors": "languages/english_2026-09_2/embeddings/alba.safetensors",
    "voice_vera.safetensors": "languages/english_2026-09_2/embeddings/vera.safetensors",
    "voice_michael.safetensors": "languages/english_2026-09_2/embeddings/michael.safetensors",
}

# Authored checkpoint config.json; byte-identical to the models.json pin.
CONFIG = {
    "model_type": "pocket_tts",
    "recipe": "english_2026-09_2",
    "flow_lm": {
        "dtype": "float32", "insert_bos_before_voice": True, "flow_type": "lsd",
        "transformer": {"d_model": 1024, "hidden_scale": 4, "max_period": 10000,
                        "num_heads": 16, "num_layers": 6},
        "flow": {"depth": 6, "dim": 512},
        "lookup_table": {"dim": 1024, "n_bins": 4000, "tokenizer": "tokenizers"},
    },
    "mimi": {
        "dtype": "float32", "sample_rate": 24000, "channels": 1, "frame_rate": 12.5,
        "inner_dim": 32, "outer_dim": 512,
        "seanet": {"dimension": 512, "channels": 1, "n_filters": 64,
                   "n_residual_layers": 1, "ratios": [6, 5, 4], "kernel_size": 7,
                   "residual_kernel_size": 3, "last_kernel_size": 3,
                   "dilation_base": 2, "pad_mode": "constant", "compress": 2},
        "transformer": {"d_model": 512, "num_heads": 8, "num_layers": 2,
                        "layer_scale": 0.01, "context": 250, "dim_feedforward": 2048,
                        "input_dimension": 512, "output_dimensions": [512],
                        "max_period": 10000.0},
        "quantizer": {"dimension": 32, "output_dimension": 512},
    },
    "generation": {"temperature": 0.3, "sampler_decode_steps": 1, "noise_clamp": None,
                   "eos_threshold": -4.0, "max_text_tokens": 50, "min_frames_before_eos": 6,
                   "tokens_per_second_estimate": 3.0, "gen_seconds_padding": 2.0,
                   "frames_after_eos_extra": 2},
}
CONFIG_JSON = json.dumps(CONFIG, indent=1, sort_keys=True)

# Recipe YAML materialized with local paths (content mirrors the pinned
# english_2026-09_2.yaml; {checkpoint} and {dtype} are formatted at prepare time).
CONFIG_YAML_TEMPLATE = """# pocket-tts recipe english_2026-09_2, materialized by the port pack with local
# weights paths. Numeric content mirrors the pinned YAML at
# github.com/kyutai-labs/pocket-tts ec569b2dd3fec4a601b391aace185791e92b3785.
weights_path: {checkpoint}/model.safetensors
default_temperature: 0.3

flow_lm:
  insert_bos_before_voice: true
  dtype: {dtype}
  flow:
    depth: 6
    dim: 512
  transformer:
    d_model: 1024
    hidden_scale: 4
    max_period: 10000
    num_heads: 16
    num_layers: 6
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: tokenizers
    tokenizer_path: {checkpoint}/tokenizer.json

mimi:
  dtype: {dtype}
  sample_rate: 24000
  inner_dim: 32
  outer_dim: 512
  channels: 1
  frame_rate: 12.5
  seanet:
    dimension: 512
    channels: 1
    n_filters: 64
    n_residual_layers: 1
    ratios:
    - 6
    - 5
    - 4
    kernel_size: 7
    residual_kernel_size: 3
    last_kernel_size: 3
    dilation_base: 2
    pad_mode: constant
    compress: 2
  transformer:
    d_model: 512
    num_heads: 8
    num_layers: 2
    layer_scale: 0.01
    context: 250
    dim_feedforward: 2048
    input_dimension: 512
    output_dimensions:
    - 512
  quantizer:
    dimension: 32
    output_dimension: 512
"""


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


# --------------------------------------------------------------------- suite

def make_suite(corpus, stage, model=KEY):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in portforge_eval.STAGES:
        raise ValueError("unrecognized corpus or stage")
    entries = {entry["name"]: entry for entry in corpus["cases"]}
    cases, labels = [], {}
    anchors = list(ANCHOR_FRAMES) if stage in ("smoke", "bringup", "perf") else []

    def add(name, key, numerical=True, **extra):
        entry = entries[key]
        if entry["voice"] not in VOICES:
            raise ValueError("corpus voice is not pinned in the registry")
        cases.append(dict(name=name, clip=key, voice=entry["voice"], numerical=numerical,
                          anchors=list(anchors), **extra))
        labels[name] = entry["label"]

    if stage == "smoke":
        add("short", corpus["selection"]["smoke"])
    elif stage in ("bringup", "perf"):
        pick = corpus["selection"]["bringup"]
        add("short", pick["short"])
        add("long", pick["long"])
        add("one_word", pick["one_word"])
        add("digits", pick["digits"])
        add("punct", pick["punct"])
        add("upper", pick["upper"])
        add("maxlen", pick["maxlen"])
        # Revisit reuses the SHORT clip on purpose: identical text, voice and
        # (clip-seeded) noise, so it must reproduce the base case bit-for-bit.
        add("revisit", pick["short"], compare_to="short", compare_rows=[0])
    else:
        full = corpus["selection"]["full"]
        for key in full:
            if key == "full-revisit":
                add("revisit", full[0], quality=True, compare_to=full[0], compare_rows=[0])
            else:
                add(key, key, quality=True)
    spec = dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"],
                corpus_sha256=CORPUS_SHA256, sample_rate=SAMPLE_RATE, frame_rate=FRAME_RATE,
                latent_dim=LATENT_DIM, voices=list(VOICES), anchor_frames=list(ANCHOR_FRAMES),
                gates=GATES.copy(), timed_repeats=portforge_eval.stage_repeats(stage), **portforge_eval.perf_fields(stage), cases=cases)
    return check_suite(spec), labels


# ------------------------------------------------------------------ helpers

def case_noise(clip, corpus_sha, rows, dim=LATENT_DIM):
    """Deterministic per-clip noise pool: one row per generated frame.

    Seeded by the corpus clip (not the suite case name) so a revisit case -
    same clip, same voice - replays the identical generation as its base case.
    """
    import numpy as np
    seed = int(hashlib.sha256((corpus_sha + ":" + clip).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(0.0, NOISE_STD, (rows, dim)).astype(np.float32)


def max_frames_for(token_count):
    gen_seconds = token_count / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING
    return math.ceil(gen_seconds * FRAME_RATE)


def max_row_nrmse(actual, expected):
    """Max per-row NRMSE over the leading axis, computed in float64."""
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite values")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    axes = tuple(range(1, a.ndim))
    error = np.sqrt(np.mean((a - b) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(b ** 2, axis=axes)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    import numpy as np
    fmax = fmax or sr / 2

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mels = hz_to_mel(freqs)
    bank = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        left, center, right = points[i], points[i + 1], points[i + 2]
        up = (mels - left) / max(center - left, 1e-12)
        down = (right - mels) / max(right - center, 1e-12)
        bank[i] = np.clip(np.minimum(up, down), 0.0, None)
    return bank


def logmel(audio, sr=SAMPLE_RATE, n_fft=1024, hop=256, n_mels=80):
    """Deterministic numpy log-mel used only to compare two audio renderings."""
    import numpy as np
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("audio must be one-dimensional")
    window = np.hanning(n_fft)
    count = 1 + max(0, (len(x) - n_fft)) // hop
    frames = (np.stack([x[i * hop:i * hop + n_fft] * window for i in range(count)])
              if count else np.zeros((0, n_fft)))
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = power @ _mel_filterbank(sr, n_fft, n_mels).T
    return np.log10(mel + 1e-10)


def audio_mel_nrmse(actual, expected):
    import numpy as np
    a, b = logmel(actual), logmel(expected)
    common = min(len(a), len(b))
    if common < 8:
        raise ValueError("audio too short for mel comparison")
    a, b = a[:common], b[:common]
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(1e-12, np.sqrt(np.mean(b ** 2))))


_STRIP = re.compile(r"[^A-Za-z0-9']")


def normalize_words(text):
    words = []
    for word in str(text).split():
        cleaned = _STRIP.sub("", word).upper()
        if cleaned:
            words.append(cleaned)
    return words


def word_error_rate(hypothesis, reference_text):
    """Word-level Levenshtein distance over normalized whitespace tokens."""
    hyp, ref = normalize_words(hypothesis), normalize_words(reference_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(ref) + 1))
    for i, word in enumerate(hyp, 1):
        current = [i]
        for j, ref_word in enumerate(ref, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != ref_word)))
        previous = current
    return previous[-1] / len(ref)


# ----------------------------------------------------------------- checkpoint

def hf_token():
    """The stored Hugging Face token, read explicitly.

    Reference jobs run with HF_HUB_DISABLE_IMPLICIT_TOKEN=1, under which even
    token=True refuses the implicit stored token; the gated repository still
    requires authentication, so read the standard location ourselves.
    """
    path = Path(os.environ.get("HF_TOKEN_PATH", Path.home() / ".cache/huggingface/token"))
    token = os.environ.get("HF_TOKEN") or (path.read_text().strip() if path.is_file() else "")
    if not token:
        raise RuntimeError("the pinned repository is gated: log in with `hf auth login` "
                           "on this host (or set HF_TOKEN) and accept the terms")
    return token


def prepare_checkpoint(directory):
    """Download the pinned gated subpaths, flatten, write authored config files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pinned = model_spec(KEY)["files"]["config.json"]
    digest, size = hashlib.sha256(CONFIG_JSON.encode()).hexdigest(), len(CONFIG_JSON)
    if digest != pinned["sha256"] or size != pinned["bytes"]:
        raise ValueError("authored config.json drifted from the models.json pin")
    (directory / "config.json").write_text(CONFIG_JSON)
    (directory / "config.yaml").write_text(CONFIG_YAML_TEMPLATE.format(
        checkpoint=str(directory.resolve()), dtype="float32"))
    from huggingface_hub import hf_hub_download
    spec, token = model_spec(KEY), hf_token()
    for flat, subpath in HF_PATHS.items():
        target = directory / flat
        if target.is_file() and target.stat().st_size == spec["files"][flat]["bytes"]:
            continue
        source = Path(hf_hub_download(repo_id=MODEL, filename=subpath, revision=REVISION, token=token))
        target.write_bytes(source.read_bytes())
    return verify_checkpoint(directory, KEY, all_files=True)


def export_voice_state(state):
    """ModelState dict -> {flat_name: numpy array} for the public inputs."""
    arrays = {}
    for module, fields in state.items():
        for field, value in fields.items():
            arrays[module.replace(".", "_") + "__" + field] = value.detach().cpu().numpy()
    return arrays


def import_voice_state(arrays, device):
    """Public input arrays -> nested ModelState for the official implementation."""
    import torch
    state = {}
    for i in range(TRANSFORMER_LAYERS):
        module = f"transformer.layers.{i}.self_attn"
        for tail in ("cache", "offset", "pad"):
            flat = f"transformer_layers_{i}_self_attn__{tail}"
            if flat in arrays:
                state.setdefault(module, {})[tail] = torch.from_numpy(
                    arrays[flat].copy()).to(device)
    if not state:
        raise ValueError("no voice-state arrays found in public inputs")
    return state


# -------------------------------------------------------------------- oracle

class NoisePool:
    def __init__(self, rows):
        self.rows, self.index = rows, 0

    def fill(self, tensor):
        import torch
        if self.index >= len(self.rows):
            raise ValueError("noise pool exhausted before generation stopped")
        row = torch.from_numpy(self.rows[self.index])
        tensor.copy_(row.to(device=tensor.device, dtype=tensor.dtype).reshape(tensor.shape))
        self.index += 1


@contextlib.contextmanager
def pooled_noise(pool):
    """Feed the official sampler the frozen pool rows, in draw order."""
    import torch
    original = torch.nn.init.normal_

    def patched(tensor, mean=0.0, std=1.0, generator=None):
        if (generator is None and tensor.numel() == LATENT_DIM and float(mean) == 0.0
                and abs(float(std) - NOISE_STD) < 1e-9):
            pool.fill(tensor)
            return tensor
        return original(tensor, mean=mean, std=std, generator=generator)

    torch.nn.init.normal_ = patched
    try:
        yield pool
    finally:
        torch.nn.init.normal_ = original


def replay_generation(model, voice_state, prepared, frames_after_eos, pool):
    """Faithful replay of the official autoregressive loop; returns latents, eos."""
    import numpy as np
    import torch
    state = copy.deepcopy(voice_state)
    token_count = prepared.shape[1]
    max_gen_len = max_frames_for(token_count)
    current_end = model._flow_lm_current_end(state)
    model._expand_kv_cache(state, sequence_length=current_end + token_count + max_gen_len)
    model._run_flow_lm_and_increment_step(model_state=state, text_tokens=prepared)
    backbone_input = torch.full((1, 1, model.flow_lm.ldim), float("NaN"),
                                device=next(model.parameters()).device, dtype=model.flow_lm.dtype)
    latents, eos_frame = [], None
    for step in range(max_gen_len):
        next_latent, is_eos = model._run_flow_lm_and_increment_step(
            model_state=state, backbone_input_latents=backbone_input)
        latents.append(next_latent[:, 0].float().cpu().numpy().reshape(-1))
        if bool(torch.any(is_eos)) and eos_frame is None and step >= MIN_FRAMES_BEFORE_EOS:
            eos_frame = step
        if eos_frame is not None and step >= eos_frame + frames_after_eos:
            break
        backbone_input = next_latent
    return np.stack(latents), (eos_frame if eos_frame is not None else -1)


def match_state_dtype(state, model):
    """Cast a voice state's float tensors to the model's parameter dtype.

    The official state importer moves to device only; a bf16 precision study
    would otherwise mix fp32 caches into bf16 layers (an upstream limitation
    this diagnostic-only benchmark resolves explicitly).
    """
    dtype = next(model.parameters()).dtype
    for module_state in state.values():
        for field, tensor in module_state.items():
            if tensor.is_floating_point():
                module_state[field] = tensor.to(dtype)
    return state


def load_tts(checkpoint, device, dtype_name="fp32"):
    """Official implementation from the pinned local checkpoint.

    The materialized YAML carries the requested dtype natively (float32 for the
    immutable oracle; bfloat16 only for the diagnostic precision study), so the
    official streaming path stays dtype-consistent end to end.
    """
    import torch
    from pocket_tts import TTSModel
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
    checkpoint = Path(checkpoint)
    yaml_name = "config.yaml" if dtype_name == "fp32" else f"config-{dtype_name}.yaml"
    yaml_path = checkpoint / yaml_name
    if not yaml_path.is_file():
        yaml_path.write_text(CONFIG_YAML_TEMPLATE.format(
            checkpoint=str(checkpoint.resolve()), dtype={"fp32": "float32",
                                                         "bf16": "bfloat16"}[dtype_name]))
    model = TTSModel.load_model(config=str(yaml_path), temp=TEMPERATURE,
                                sampler_decode_steps=DECODE_STEPS, noise_clamp=None,
                                eos_threshold=EOS_THRESHOLD)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    # Freeze parameters: the official streaming pipeline spawns decoder threads that
    # do not inherit no_grad/inference_mode thread-local state, so graphs are never built.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device=device, dtype=dtype).eval()
    if dtype_name == "bf16":
        # The official FlowLM.forward upcasts the transformer output to fp32 before
        # the EOS/flow heads, which makes an all-bf16 execution impossible; the AR
        # loop would also feed the fp32 latents back into bf16 layers. For this
        # DIAGNOSTIC study only, run the official forward with that single upcast
        # removed so the whole pipeline is natively bf16. The FP32 oracle is never
        # affected. (Mirrors pocket_tts 3.3.0 FlowLMModel.forward, MIT.)
        from functools import partial as _partial
        from pocket_tts.models.flow_lm import lsd_decode, ot_decode

        def forward_bf16(self, sequence, text_embeddings, model_state, sampler_decode_steps,
                         temp, noise_clamp, eos_threshold):
            sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
            input_ = self.input_linear(sequence)
            transformer_out = self.backbone(input_, text_embeddings, sequence, model_state=model_state)
            assert sampler_decode_steps > 0
            transformer_out = transformer_out[:, -1]
            out_eos = self.out_eos(transformer_out) > eos_threshold
            noise_shape = transformer_out.shape[:-1] + (self.ldim,)
            std = temp ** 0.5
            noise = torch.empty(noise_shape, dtype=transformer_out.dtype, device=transformer_out.device)
            if noise_clamp is None:
                torch.nn.init.normal_(noise, mean=0.0, std=std)
            else:
                torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-noise_clamp, b=noise_clamp)
            conditioned_flow = _partial(self.flow_net, transformer_out)
            decode = ot_decode if self.flow_type == "flow_matching" else lsd_decode
            return decode(conditioned_flow, noise, sampler_decode_steps), out_eos

        import types
        model.flow_lm.forward = types.MethodType(forward_bf16, model.flow_lm)
        # The streaming codec states are initialized in fp32 upstream and its convs do
        # not promote, so the Mimi decode stays fp32 and casts the bf16 latents at the
        # boundary. The study therefore measures bf16 drift in the AR transformer +
        # flow head; the codec renders those latents in fp32.
        model.mimi = model.mimi.to(torch.float32)
        original_decode = model.mimi.decode_from_latent

        def _decode_fp32(latent, mimi_state, _original=original_decode):
            return _original(latent.float(), mimi_state)

        model.mimi.decode_from_latent = _decode_fp32
    return model


def voice_states(model, checkpoint):
    return {name: model.get_state_for_audio_prompt(str(Path(checkpoint) / f"voice_{name}.safetensors"))
            for name in VOICES}


def prepare_case_text(model, text):
    """Official single-chunk text preparation; also validates the envelope."""
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    chunks = split_into_best_sentences(model.flow_lm.conditioner.tokenizer, text,
                                       MAX_TEXT_TOKENS, False, False)
    if len(chunks) != 1:
        raise ValueError("corpus text escapes the single-chunk envelope: " + text[:60])
    prepared_text, guess = prepare_text_prompt(chunks[0], False, False)
    frames_after_eos = guess + FRAMES_AFTER_EOS_EXTRA
    prepared = model.flow_lm.conditioner.prepare(prepared_text)
    if prepared.shape[1] > MAX_TEXT_TOKENS:
        raise ValueError("prepared text exceeds the token envelope")
    return prepared, frames_after_eos


def run_case(model, voice_state, text, prepared, frames_after_eos, noise, want_audio=True):
    """Both oracle passes: official audio + replayed latents/EOS on one pool."""
    import numpy as np
    audio = None
    pool_a, pool_b = NoisePool(noise), NoisePool(noise)
    if want_audio:
        with pooled_noise(pool_a):
            audio = model.generate_audio(copy.deepcopy(voice_state), text,
                                         max_tokens=MAX_TEXT_TOKENS,
                                         frames_after_eos=frames_after_eos)
        audio = audio.detach().float().cpu().numpy().reshape(-1)
    with pooled_noise(pool_b):
        latents, eos_frame = replay_generation(model, voice_state, prepared, frames_after_eos, pool_b)
    # Both passes draw one sampler row for the text prefill (output discarded) and
    # one row per computed frame, in the same order.
    if pool_a.index and pool_a.index != pool_b.index:
        raise ValueError("official audio pass and latent replay consumed different noise rows")
    rendered = len(latents) * SAMPLES_PER_FRAME - (SAMPLES_PER_FRAME if eos_frame != -1 else 0)
    if want_audio and abs(len(audio) - rendered) > SAMPLES_PER_FRAME:
        raise ValueError("official audio length disagrees with replayed latent count")
    return latents, eos_frame, audio


def reference(args):
    """Reference host: the official TTSModel's latents, EOS frame and audio (FP32 oracle or benchmark).

    The gated checkpoint, the voice-state export and the pooled-noise replay are this pack's own;
    run_case's audio may differ by ~1e-6 between identical runs (see below), so only latents and EOS
    go through run.time's exact repeat check and the audio is compared with a tolerance here.
    """
    import numpy as np
    run = portforge_eval.ReferenceRun(args, globals())
    if not run.benchmark and run.precision != "fp32":
        raise ValueError("the immutable correctness oracle is FP32")
    torch, device = run.torch()
    run.progress.mark("prepare_checkpoint", initializing=True)
    run.started = time.perf_counter()
    checkpoint = run.root / "checkpoint"
    run.receipt = prepare_checkpoint(checkpoint)
    run.receipt["model_manifest_sha256"] = run.identity["model_manifest_sha256"]
    run.progress.mark("load_model", initializing=True)
    model = load_tts(checkpoint, device, run.precision)
    voices = voice_states(model, checkpoint)
    run.publish_config((checkpoint / "config.json").read_text())
    run.loaded(parameter_dtypes=[{"fp32": "float32", "bf16": "bfloat16"}[run.precision]],
               policy=dict(noise="frozen public pool rows fed via torch.nn.init.normal_ patch, frame order",
                           latents="official autoregressive loop replay over the same pool",
                           audio="official TTSModel.generate_audio, first 5 ms faded in as upstream"))
    if run.benchmark:
        prepared_inputs = np.load(run.public / "inputs.npz", allow_pickle=False)
    else:
        for name in VOICES:
            for flat, array in export_voice_state(voices[name]).items():
                run.inputs[f"voice__{name}__{flat}"] = array
        entries = {c["name"]: c for c in run.corpus["cases"]}
    with torch.no_grad():
        for case in run.cases():
            arrays = run.prepared(case)
            if arrays is None:
                entry = entries[case["clip"]]
                prepared, frames_after_eos = prepare_case_text(model, entry["text"])
                text, tokens = entry["text"], int(prepared.shape[1])
                noise = case_noise(case["clip"], CORPUS_SHA256, max_frames_for(tokens) + 8)
                voice = voices[case["voice"]]
            else:
                prepared = torch.from_numpy(arrays["text_tokens"]).long().reshape(1, -1).to(device)
                text, frames_after_eos = str(arrays["text"]), int(arrays["frames_after_eos"])
                noise = arrays["noise"]
                voice = import_voice_state(
                    {k.split("__", 2)[2]: v for k, v in prepared_inputs.items()
                     if k.startswith("voice__" + case["voice"] + "__")}, device)
                voice = match_state_dtype(voice, model)
            results = []

            def generate():
                results.append(run_case(model, voice, text, prepared, frames_after_eos, noise))
                return results[-1]

            outputs = run.time(case, generate, lambda result: {"latents": result[0],
                                                               "eos_frame": np.int64(result[1])})
            # Latents and EOS are bit-deterministic on the fixed noise pool. The
            # official decoder thread batches frames by wall-clock timing, so its
            # audio may reassociate floats at the ~1e-6 level between identical runs.
            audio = results[0][2]
            if any(not np.allclose(audio, other, rtol=0.0, atol=1e-4) for _, _, other in results[1:]):
                raise ValueError("reference nondeterminism")
            latents, eos_frame = outputs["latents"], int(outputs["eos_frame"])
            outputs["audio"] = audio
            run.meta["cases"][-1].update(audio_seconds=(len(audio) / SAMPLE_RATE if audio is not None else None),
                                         frames=len(latents), eos_frame=eos_frame,
                                         text_tokens=int(prepared.shape[1]))
            if run.benchmark:
                # The benchmark's own replay is its anchor prediction (same computation).
                for key in arrays:
                    if key.startswith("anchor_"):
                        outputs[key] = latents[int(key.rsplit("_", 1)[1])].copy()
                run.record(case, None, outputs)
                continue
            inputs = {"text_tokens": prepared[0].long().cpu().numpy(), "text": np.array(entry["text"]),
                      "frames_after_eos": np.int64(frames_after_eos), "max_frames": np.int64(max_frames_for(tokens)),
                      "noise": noise}
            for t in case["anchors"]:
                if t < len(latents):
                    inputs[f"anchor_{t}"] = latents[:t].copy()
            run.record(case, inputs, outputs)
    return run.finish()


# Candidate (TT sandbox): portforge_eval.run_candidate times forward_case; probe_case runs the
# teacher-forced anchors once per case, untimed.
# Learned ops that never run on the host CPU. Host matmuls are allowed only without
# model weights (portforge_eval.HostComputeGuard), e.g. RoPE angles from positions.
FORBIDDEN_HOST_OPS = ("aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                      "aten.softmax.", "aten.gelu.", "aten.silu.", "aten.scaled_dot_product",
                      "aten.conv1d.", "aten.conv2d.", "aten.conv_transpose1d.", "aten.convolution.")
ROWS = None  # batch 1: one item per case


def case_inputs(inputs, case, spec):
    """The case's arrays plus its voice's ModelState arrays under "voice.<flat name>" keys."""
    prefix = "voice__" + case["voice"] + "__"
    voice = {"voice." + key[len(prefix):]: inputs[key] for key in inputs.files if key.startswith(prefix)}
    return {**portforge_eval.case_arrays(inputs, case), **voice}


def supply(inputs):
    """Copies for the backend (mutation is detected against the originals) and its voice dict."""
    supplied = {key: value.copy() for key, value in inputs.items()}
    return supplied, {key[len("voice."):]: value for key, value in supplied.items() if key.startswith("voice.")}


def forward_case(backend, case, inputs, spec, config):
    """One timed generation: tokens + voice + noise -> latents, EOS frame and audio."""
    import numpy as np
    supplied, voice = supply(inputs)
    outputs = backend.generate(supplied["text_tokens"], voice, supplied["noise"],
                               int(supplied["frames_after_eos"]), int(supplied["max_frames"]))
    latents = np.asarray(outputs["latents"], dtype=np.float32)
    if latents.ndim != 2 or latents.shape[1] != LATENT_DIM or len(latents) == 0:
        raise ValueError("generate must return latents [T, 32]")
    eos = int(outputs["eos_frame"])
    if eos != -1 and not 0 <= eos < int(supplied["max_frames"]):
        raise ValueError("eos_frame outside the case frame budget")
    audio = np.asarray(outputs["audio"], dtype=np.float32).reshape(-1)
    # The official decoder renders every computed frame except the one that
    # triggers the stop (when EOS fires; a capped run renders all frames).
    rendered = len(latents) * SAMPLES_PER_FRAME - (SAMPLES_PER_FRAME if eos != -1 else 0)
    if abs(len(audio) - rendered) > 192:
        raise ValueError("audio length must match the rendered latent frames")
    return {"latents": latents, "eos_frame": np.int64(eos), "audio": audio}, supplied


def probe_case(backend, case, inputs, spec, config):
    """Teacher-forced anchors (development stages): the next latent after each oracle prefix."""
    import numpy as np
    supplied, voice = supply(inputs)
    arrays = {}
    for key in sorted(k for k in inputs if k.startswith("anchor_")):
        t = int(key.rsplit("_", 1)[1])
        # noise[0] is consumed by the text prefill (drawn and discarded, as the
        # official forward always samples); latent f is produced by noise[f + 1].
        row = np.asarray(backend.predict_latent(supplied["text_tokens"], voice,
                                                supplied[key], supplied["noise"][t + 1]),
                         dtype=np.float32).reshape(-1)
        if row.shape != (LATENT_DIM,):
            raise ValueError("predict_latent must return one 32-dim row")
        arrays[key] = row
    return arrays, supplied


def candidate(args):
    return portforge_eval.run_candidate(args, globals())


# -------------------------------------------------------------------- assess

def transcribe(audios):
    """Greedy parakeet TDT transcription at 16 kHz; the assessor's scoring ASR."""
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForTDT, AutoProcessor
    processor = AutoProcessor.from_pretrained(ASR_MODEL, revision=ASR_REVISION, trust_remote_code=False)
    model = AutoModelForTDT.from_pretrained(ASR_MODEL, revision=ASR_REVISION, trust_remote_code=False).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    texts = []
    with torch.inference_mode():
        for audio in audios:
            wave = resample_poly(np.asarray(audio, dtype=np.float64), 2, 3).astype(np.float32)
            feats = processor(wave, sampling_rate=16000, return_tensors="pt")
            out = model.generate(input_features=feats.input_features.to(device).float(),
                                 attention_mask=feats.attention_mask.to(device),
                                 return_dict_in_generate=True)
            texts.append(processor.batch_decode(out.sequences, skip_special_tokens=True)[0])
    return texts


def output_keys(case, spec):
    """The arrays the oracle holds per case; the candidate adds its anchor predictions."""
    return ("latents", "eos_frame", "audio")


def check_case(check, case, a):
    name, actual, tol = case["name"], a.actual, GATES["eos_frame_tolerance"]
    latents = actual[name + "__latents"]
    eos = int(actual[name + "__eos_frame"])
    audio = actual[name + "__audio"]
    o_latents = a.expected[name + "__latents"]
    o_eos = int(a.expected[name + "__eos_frame"])
    o_audio = a.expected[name + "__audio"]
    if latents.dtype.kind != "f" or audio.dtype.kind != "f":
        raise ValueError("latents/audio must be floating point")
    check["frames"], check["reference_frames"] = len(latents), len(o_latents)
    check["eos_frame"], check["reference_eos_frame"] = eos, o_eos
    if -1 in (eos, o_eos):
        if eos != o_eos:
            raise ValueError(f"EOS runaway: candidate {eos} vs oracle {o_eos} (tolerance {tol})")
    elif abs(eos - o_eos) > tol:
        raise ValueError(f"EOS frame differs by {abs(eos - o_eos)} > {tol}")
    common = min(len(latents), len(o_latents))
    if common < 4:
        raise ValueError("no meaningful latent overlap")
    error = max_row_nrmse(latents[:common], o_latents[:common])
    check["free_run_max_row_nrmse"] = error
    if error > GATES["free_run_nrmse"]:
        raise ValueError(f"free-run latent NRMSE exceeds {GATES['free_run_nrmse']}")
    ratio = min(len(audio), len(o_audio)) / max(1, max(len(audio), len(o_audio)))
    check["audio_length_ratio"] = ratio
    if ratio < 0.85:
        raise ValueError("audio length ratio below 0.85")
    mel_error = audio_mel_nrmse(audio, o_audio)
    check["audio_mel_nrmse"] = mel_error
    if mel_error > GATES["audio_mel_nrmse"]:
        raise ValueError(f"audio mel NRMSE exceeds {GATES['audio_mel_nrmse']}")
    worst = None
    for key in a.notes["anchors"].get(name, []):
        t = int(key.rsplit("_", 1)[1])
        if key not in actual or t >= len(o_latents):
            raise ValueError("missing anchor prediction")
        error = max_row_nrmse(actual[key].reshape(1, -1), o_latents[t].reshape(1, -1))
        worst = error if worst is None else max(worst, error)
    if worst is not None:
        check["anchor_max_row_nrmse"] = worst
        if worst > GATES["nrmse"]:
            raise ValueError(f"teacher-forced anchor NRMSE exceeds {GATES['nrmse']}")
    if "compare_to" in case:
        base = actual[case["compare_to"] + "__latents"]
        common = min(len(latents), len(base))
        error = max_row_nrmse(latents[:common], base[:common])
        check["revisit_max_row_nrmse"] = error
        if (error > GATES["behavior_max_nrmse"]
                or int(actual[case["compare_to"] + "__eos_frame"]) != eos):
            raise ValueError("revisit changed semantics")
    a.timing(check, case)
    audio_seconds = len(audio) / SAMPLE_RATE
    check.update(audio_seconds=audio_seconds, real_time_factor=audio_seconds / check["p50_seconds"])


def quality(a):
    """Full stage: WER of the scoring ASR on candidate audio, relative to the same scoring of the oracle's."""
    labels, quality = a.labels(), {}
    for case in a.spec["cases"]:
        if not case.get("quality") or case.get("compare_to"):
            continue
        name = case["name"]
        try:
            cand_text, ref_text = transcribe([a.actual[name + "__audio"], a.expected[name + "__audio"]])
        except Exception as exc:  # assessor environment issue, not a pass
            a.failures.append(name + ": scoring ASR failed (" + str(exc)[:120] + ")")
            continue
        try:
            candidate_wer = word_error_rate(cand_text, labels[name])
            reference_wer = word_error_rate(ref_text, labels[name])
            if not math.isfinite(candidate_wer) or not math.isfinite(reference_wer):
                raise ValueError("nonfinite WER")
            inflation = candidate_wer - reference_wer
            quality[name] = dict(wer=candidate_wer, reference_wer=reference_wer,
                                 wer_inflation=inflation, hypothesis=cand_text,
                                 reference_hypothesis=ref_text)
            if inflation > GATES["max_wer_inflation"]:
                a.failures.append(name + f": WER inflation exceeds {GATES['max_wer_inflation']:.0%}")
        except (ValueError, TypeError, KeyError) as exc:
            a.failures.append(name + ": " + str(exc))
    return quality


def verdict_fields(a):
    """max_nrmse is the teacher-forced anchor error; the free-run error is reported beside it."""
    def worst(field):
        return max((c[field] for c in a.checks if c.get(field) is not None), default=None)
    return dict(max_nrmse=worst("anchor_max_row_nrmse"), max_free_run_nrmse=worst("free_run_max_row_nrmse"))


def candidate_keys(case, a):
    """The candidate's anchor predictions, one per anchor prefix in inputs.npz, scored against oracle latent rows."""
    anchors = a.notes.get("anchors")
    if anchors is None:
        anchors = a.notes["anchors"] = {}
        for key in sorted(k for k in a.inputs.files if "__anchor_" in k):
            anchors.setdefault(key.split("__")[0], []).append(key)
    return anchors.get(case["name"], [])


def assess(args):
    return portforge_eval.run_assess(args, globals())


if __name__ == "__main__":
    portforge_eval.main(globals())
