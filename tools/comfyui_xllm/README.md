# xLLM Server Mode for ComfyUI

Run MiniMax-H3's Python T2VA pipeline on a persistent xLLM Ascend server and return
an audio-video `VIDEO` to ComfyUI. The plugin does not load the DiT or the VAEs,
start xLLM workers, or require a native xLLM installation in ComfyUI's environment.

## Install

Copy the entire `tools/comfyui_xllm` directory into `ComfyUI/custom_nodes/comfyui_xllm`
and restart ComfyUI. Use a ComfyUI version with the V3 API (`comfy_api.latest`),
native `VIDEO`, and MiniMax-H3 text encoding. `torch` and `aiohttp` are already ComfyUI dependencies.

Start a matching xLLM server separately using the
[MiniMax-H3 deployment guide](../../docs/src/content/docs/en/cookbook/diffusion_models/minimax_h3/minimax_h3.md).
For the Python pipeline, select `model_impl=python`, TP2/SP2/VAE4, CFG1/DP1,
batch 1, `dit_h3_task_type=fl2va`, `dit_cache_policy=None`, and disable graph,
sparse attention and rolling load. The server's model ID must match the node.
The base URL is the root, for example `http://127.0.0.1:8000`, without `/v1`.

## Generate from text

1. Use ComfyUI's **Load CLIP** with `type=minimax` and the matching **MiniMax-H3
   video text-encoder** weights. These are the H3 Qwen3-VL-32B encoder weights
   truncated to the required layer, not an arbitrary CLIP/Qwen checkpoint or the
   H3-Image encoder. Select CPU if the ComfyUI host has no supported accelerator.
   Provision RAM for the encoder weights and workspace; the conditioning-file
   route allows text encoding on another machine.
2. Connect it to **xLLM MiniMax-H3 Text Encode** and enter the prompt.
3. Connect that conditioning and **xLLM Server** to **xLLM Generate MiniMax-H3**.
4. Connect `VIDEO` to **Save Video**. **Get Video Components** can extract frames,
   audio and FPS for later processing.

The text node uses ComfyUI's actual H3 tokenizer and encoder, forwarding its
`minimax_token_tags` with the embeddings. It never invents tags or substitutes
another model's embeddings. Only text-only, unscheduled conditioning is supported.
ComfyUI owns text encoding; xLLM owns the complete denoising and decoding process.

## Generate from existing conditioning

Put a trusted `conditioning.pt` in `ComfyUI/input/` and use **xLLM MiniMax-H3
Conditioning Loader** instead of the text-encoding node. The file contains:

- `prompt`: the original string used by the H3 encoder;
- `prompt_embed`: floating point `[L, 5120]` or `[1, L, 5120]`;
- `prompt_token_tags`: int64 `[L]`, all ones for the current Python T2VA path.

Loading uses `torch.load(..., weights_only=True)` on CPU. Changing file contents
invalidates the loader's cache. This route is useful for comparing a known
conditioning against a direct HTTP request; it does not perform text encoding.

## Parameters and results

The initial settings are 1344×768, 124 frames, 24 FPS, 50 steps and seed 42.
Each request produces one video with 32 kHz audio. xLLM aligns the dimensions and
frame count; inspect the node's `metadata` output for the actual returned values.
The current boundary requires `frames / fps >= 5` and `(frames + 16) / fps <= 15`.

The plugin exposes only controls used by the current Python H3 pipeline. It does
not expose reference images/audio/video, arbitrary samplers, negative prompts or
CFG controls. Those need corresponding backend support before adding widgets.

The HTTP client uses `/v1/video/generation` and base64 MP4 output. It preserves
server errors and never retries a generation automatically. Cancel stops the
client waiting; an already running server generation may continue. There is no
per-step progress API in this xLLM endpoint.

ComfyUI caches node results. Change the seed to generate a new sample. Clear the
ComfyUI execution cache after changing weights or backend implementation behind
an unchanged endpoint/model ID. Temporary MP4 files remain available to downstream
nodes and cached `VIDEO` objects; use Save Video for durable output.

## Example API workflows

`workflows/minimax_h3_conditioning_api.json` and `workflows/minimax_h3_text_to_video_api.json` are
ComfyUI **API prompt dictionaries**, for submission in the `prompt` field to
ComfyUI's `/prompt` endpoint. They are not UI-exported canvas layouts. Set the
server URL/model ID and, for text input, replace the example `clip_name` with the
actual filename offered by Load CLIP. Build the same node chain in the UI for
interactive use.
