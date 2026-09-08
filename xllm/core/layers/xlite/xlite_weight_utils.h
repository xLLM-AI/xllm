/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// StateDict -> xlite XModel weight binding. Fuses qkv/gate_up via torch::cat.

#pragma once

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>  // at_npu::native::npu_format_cast (NZ)
#include <xlite/xlite.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/xlite/xlite_init_utils.h"

namespace xllm::xlite {

class XliteWeightUtils {
 public:
  // .to(device) and retain in storages (InitXTensor only stores data_ptr).
  static const torch::Tensor& ToDevice(const torch::Tensor& t,
                                       const std::string& name,
                                       const torch::Device& device,
                                       std::vector<torch::Tensor>& storages) {
    CHECK(t.defined()) << "[xlite] missing weight: " << name;
    storages.push_back(t.to(device, /*non_blocking=*/false));
    return storages.back();
  }

  // W8A8 INT8 weight NZ pre-convert (npu_format_cast). INT8 matmul only; call
  // after ToDevice. In-place replaces storages.back() (avoids ND+NZ double
  // copy). BF16/deqScale excluded.
  static const torch::Tensor& CastNz(const torch::Tensor& t,
                                     const std::string& name,
                                     std::vector<torch::Tensor>& storages) {
    CHECK(t.defined()) << "[xlite] CastNz undefined: " << name;
    CHECK(!storages.empty())
        << "[xlite] CastNz: storages empty, expected ND tensor from ToDevice: "
        << name;
    storages.back() =
        at_npu::native::npu_format_cast(t.contiguous(), ACL_FORMAT_FRACTAL_NZ);
    return storages.back();
  }

  // TP-shard a weight along dim (returns rank-th shard). GQA: pass adjusted
  // kv_tp_rank/kv_tp_size when n_kv_heads < tp_size.
  static torch::Tensor Shard(const StateDict& sd,
                             const std::string& name,
                             int64_t dim,
                             int32_t tp_rank,
                             int32_t tp_size) {
    torch::Tensor t = sd.get_sharded_tensor(name, dim, tp_rank, tp_size);
    CHECK(t.defined()) << "[xlite] missing weight: " << name;
    return t;
  }

  // W8A8 deqScale transform: FP32[N] -> FP32[2N] (fixpipe uint64 layout, scale
  // at even indices). Applies to weight_scale (BF16 [out,1]) and deq_scale (F32
  // [out]); csrc expects this layout.
  static torch::Tensor TransformDeqScale(const torch::Tensor& scale) {
    torch::Tensor fp32 = scale.to(torch::kFloat32).view({-1}).contiguous();
    int64_t n = fp32.size(0);
    torch::Tensor out = torch::zeros({n * 2}, fp32.options());
    out.slice(0, 0, 2 * n, 2) = fp32;  // stride 2: FP32 at even, 0 at odd
    return out;
  }

  // Whether weight is quantized (INT8). BF16 models skip quant fields; W8A8
  // loads them.
  static bool IsQuant(const torch::Tensor& w) {
    return w.defined() && w.scalar_type() == torch::kInt8;
  }

  // embed/norm/head + per-layer attn (qkv fuse, o_proj, qk_norm) + mlpNorm.
  // TP: embed/head vocab dim0, q/k/v out dim0, o_proj in dim1.
  static void LoadAttnAndEmbed(const StateDict& sd,
                               XModel& m,
                               const XModelConfig& cfg,
                               const ModelArgs& args,
                               const ParallelArgs& pa,
                               const torch::Device& device,
                               std::vector<torch::Tensor>& storages) {
    auto dev = [&](const torch::Tensor& t,
                   const std::string& name) -> const torch::Tensor& {
      return ToDevice(t, name, device, storages);
    };
    // W8A8 norm weight is FP32 (ATB format); csrc norm kernel reads BF16 ->
    // cast to avoid garbage. BF16 models: no-op. Norm bias handled separately.
    auto devBf16 = [&](const torch::Tensor& t,
                       const std::string& name) -> const torch::Tensor& {
      if (t.scalar_type() == torch::kBFloat16) {
        return dev(t, name);
      }
      return dev(t.to(torch::kBFloat16).contiguous(), name);
    };

    const int32_t tp = static_cast<int32_t>(XliteTpSize(pa));
    const int32_t rank = static_cast<int32_t>(XliteTpRank(pa));
    // GQA: n_kv_heads < tp_size -> kv weights replicate across tp/n_kv_heads
    // ranks.
    const uint32_t nKvHeads = cfg.nKvHeads;
    int32_t kv_tp = tp;
    int32_t kv_rank = rank;
    if (tp > 1 && nKvHeads > 0 && static_cast<uint32_t>(tp) > nKvHeads) {
      const int32_t repeat = tp / static_cast<int32_t>(nKvHeads);
      kv_tp = static_cast<int32_t>(nKvHeads);
      kv_rank = rank / repeat;
    }

    LOG(INFO) << "[xlite] LoadAttnAndEmbed: tp=" << tp << " rank=" << rank
              << " kv_tp=" << kv_tp << " kv_rank=" << kv_rank;

    // embed: vocab dim0.
    // dev() once and rebind embed so the tie branch can share it (zero-copy).
    torch::Tensor embed =
        (tp > 1) ? Shard(sd, "model.embed_tokens.weight", 0, rank, tp)
                 : sd.get_tensor("model.embed_tokens.weight");
    embed = dev(embed, "model.embed_tokens.weight");
    InitXTensor(m.embed, embed);
    // norm: not sharded.
    InitXTensor(
        m.norm,
        devBf16(sd.get_tensor("model.norm.weight"), "model.norm.weight"));
    // W8A8 norm bias is FP32 (ATB format); csrc norm kernel reads BF16 -> must
    // cast to avoid byte reinterpretation garbage. BF16 models (RMSNorm) have
    // no bias -> skip.
    torch::Tensor normBias = sd.get_tensor("model.norm.bias");
    if (normBias.defined()) {
      torch::Tensor b = normBias.to(torch::kBFloat16).contiguous();
      InitXTensor(m.normBias, dev(b, "model.norm.bias"));
    }
    // head: vocab dim0.
    torch::Tensor head;
    if (args.tie_word_embeddings()) {
      // tie: reuse embed's device storage (InitXTensor is zero-copy). Avoids a
      // second .to(device) of embed (~300MB for Qwen3-0.6B).
      head = embed;
      InitXTensor(m.head, embed);
    } else {
      head = (tp > 1) ? Shard(sd, "lm_head.weight", 0, rank, tp)
                      : sd.get_tensor("lm_head.weight");
      // GLM-5.2-W8A8: lm_head.weight is FP32; xlite XliteOpMatmul only supports
      // BF16xFP32->FP32. ForwardGetLogits localOutput is BF16 (hiddenState
      // dtype) -> cast to BF16. GLM-5.1 BF16 model: no-op.
      if (head.scalar_type() != torch::kBFloat16) {
        head = head.to(torch::kBFloat16).contiguous();
      }
      InitXTensor(m.head, dev(head, "lm_head.weight"));
    }
    if (tp > 1) {
      LOG(INFO) << "[xlite] shard rank=" << rank << ": embed=" << embed.sizes()
                << " head=" << head.sizes();
    }

    for (uint32_t i = 0; i < cfg.nLayers; ++i) {
      const std::string L = "model.layers." + std::to_string(i) + ".";
      // attnNorm: per-token, not sharded.
      InitXTensor(m.attnNorm[i],
                  devBf16(sd.get_tensor(L + "input_layernorm.weight"),
                          L + "input_layernorm.weight"));
      // W8A8 q/k_norm bias; BF16 models have none. Cast to BF16 (same as
      // normBias).
      torch::Tensor attnNormBias = sd.get_tensor(L + "input_layernorm.bias");
      if (attnNormBias.defined()) {
        torch::Tensor b = attnNormBias.to(torch::kBFloat16).contiguous();
        InitXTensor(m.attnNormBias[i], dev(b, L + "input_layernorm.bias"));
      }

      // q/k/v: out dim0. q by nHeads/TP, kv by nKvHeads/TP (GQA-adjusted).
      torch::Tensor q, k, v;
      if (tp > 1) {
        q = Shard(sd, L + "self_attn.q_proj.weight", 0, rank, tp);
        k = Shard(sd, L + "self_attn.k_proj.weight", 0, kv_rank, kv_tp);
        v = Shard(sd, L + "self_attn.v_proj.weight", 0, kv_rank, kv_tp);
      } else {
        q = sd.get_tensor(L + "self_attn.q_proj.weight");
        k = sd.get_tensor(L + "self_attn.k_proj.weight");
        v = sd.get_tensor(L + "self_attn.v_proj.weight");
      }
      // host cat then one H2D (vs dev each shard + device cat).
      // BF16 (NO_QUANT): [out, in] no transpose. W8A8 (STATIC):
      // isTransposed=true, csrc expects [in, out] -> .t(). deqScale/quantBias
      // per-out-channel unaffected.
      torch::Tensor qkv = torch::cat({q, k, v}, /*dim=*/0)
                              .contiguous();  // host [q+k+v, hidden]
      if (IsQuant(q)) {
        qkv = qkv.t().contiguous();  // W8A8: [hidden, q+k+v] = [in, out]
      }
      qkv = dev(qkv, L + "self_attn.qkv (mhaQKV)");
      // W8A8 INT8 NZ: quantAttnWeightNz.
      if (IsQuant(q) && cfg.quantAttnWeightNz) {
        qkv = CastNz(qkv, L + "self_attn.qkv (mhaQKV) NZ", storages);
      }
      InitXTensor(m.mhaQKV[i].weight, qkv);
      storages.push_back(qkv);

      // W8A8 STATIC quant: I8 weight +
      // inputScale/inputOffset/deqScale/quantBias. BF16 skips.
      // inputScale/inputOffset: per-tensor [1], same for q/k/v (take q's), not
      // sharded.
      if (IsQuant(q)) {
        torch::Tensor iscale =
            sd.get_tensor(L + "self_attn.q_proj.input_scale");
        torch::Tensor ioffset =
            sd.get_tensor(L + "self_attn.q_proj.input_offset");
        if (iscale.defined()) {
          // ATB input_scale is scale; csrc expects 1/scale (reciprocal) as BF16
          // [hidden]. Cast to BF16 + repeat to hiddenSize (per-tensor; csrc
          // reads [hidden] per-channel).
          torch::Tensor recip = (1.0f / iscale.to(torch::kFloat32))
                                    .to(torch::kBFloat16)
                                    .contiguous();
          recip = recip.repeat({static_cast<int64_t>(cfg.hiddenSize)});
          InitXTensor(m.mhaQKV[i].inputScale,
                      dev(recip, L + "q_proj.input_scale_reciprocal"));
        }
        if (ioffset.defined()) {
          // input_offset is zero-point (not reciprocal); same [hidden] repeat
          // as inputScale.
          torch::Tensor off =
              ioffset.to(torch::kBFloat16)
                  .contiguous()
                  .repeat({static_cast<int64_t>(cfg.hiddenSize)});
          InitXTensor(m.mhaQKV[i].inputOffset,
                      dev(off, L + "q_proj.input_offset"));
        }
        // quantBias: cat(q,k,v) + TP shard dim0 (same as weight).
        torch::Tensor qb_q, qb_k, qb_v;
        if (tp > 1) {
          qb_q = Shard(sd, L + "self_attn.q_proj.quant_bias", 0, rank, tp);
          qb_k =
              Shard(sd, L + "self_attn.k_proj.quant_bias", 0, kv_rank, kv_tp);
          qb_v =
              Shard(sd, L + "self_attn.v_proj.quant_bias", 0, kv_rank, kv_tp);
        } else {
          qb_q = sd.get_tensor(L + "self_attn.q_proj.quant_bias");
          qb_k = sd.get_tensor(L + "self_attn.k_proj.quant_bias");
          qb_v = sd.get_tensor(L + "self_attn.v_proj.quant_bias");
        }
        if (qb_q.defined()) {
          torch::Tensor qb =
              torch::cat({qb_q, qb_k, qb_v}, /*dim=*/0).contiguous();
          InitXTensor(m.mhaQKV[i].quantBias, dev(qb, L + "qkv.quant_bias"));
        }
        // deqScale: cat(q,k,v) -> TransformDeqScale (FP32[2N] fixpipe) -> H2D.
        torch::Tensor ds_q, ds_k, ds_v;
        if (tp > 1) {
          ds_q = Shard(sd, L + "self_attn.q_proj.deq_scale", 0, rank, tp);
          ds_k = Shard(sd, L + "self_attn.k_proj.deq_scale", 0, kv_rank, kv_tp);
          ds_v = Shard(sd, L + "self_attn.v_proj.deq_scale", 0, kv_rank, kv_tp);
        } else {
          ds_q = sd.get_tensor(L + "self_attn.q_proj.deq_scale");
          ds_k = sd.get_tensor(L + "self_attn.k_proj.deq_scale");
          ds_v = sd.get_tensor(L + "self_attn.v_proj.deq_scale");
        }
        if (ds_q.defined()) {
          torch::Tensor ds = TransformDeqScale(
              torch::cat({ds_q, ds_k, ds_v}, /*dim=*/0).contiguous());
          InitXTensor(m.mhaQKV[i].deqScale, dev(ds, L + "qkv.deq_scale"));
        }
      }

      // o_proj: in dim1.
      torch::Tensor o =
          (tp > 1) ? Shard(sd, L + "self_attn.o_proj.weight", 1, rank, tp)
                   : sd.get_tensor(L + "self_attn.o_proj.weight");
      // W8A8 STATIC quant: I8 weight + inputScale/inputOffset/quantBias/
      // deqScale. BF16 skips.
      bool oQuant = IsQuant(o);
      if (oQuant) {
        o = o.t().contiguous();  // W8A8: [heads*headDim, hidden] = [in, out]
      }
      o = dev(o, L + "self_attn.o_proj.weight");
      if (oQuant && cfg.quantAttnWeightNz) {
        o = CastNz(o, L + "self_attn.o_proj.weight NZ", storages);
      }
      InitXTensor(m.attnOut[i].weight, o);
      storages.push_back(o);
      if (oQuant) {
        uint32_t nLocalHeads = cfg.nHeads / tp;
        int64_t oInputDim = static_cast<int64_t>(nLocalHeads) * cfg.headDim;
        torch::Tensor iscale =
            sd.get_tensor(L + "self_attn.o_proj.input_scale");
        torch::Tensor ioffset =
            sd.get_tensor(L + "self_attn.o_proj.input_offset");
        if (iscale.defined()) {
          torch::Tensor recip = (1.0f / iscale.to(torch::kFloat32))
                                    .to(torch::kBFloat16)
                                    .contiguous();
          recip = recip.repeat({oInputDim});
          InitXTensor(m.attnOut[i].inputScale,
                      dev(recip, L + "o_proj.input_scale_reciprocal"));
        }
        if (ioffset.defined()) {
          torch::Tensor off =
              ioffset.to(torch::kBFloat16).contiguous().repeat({oInputDim});
          InitXTensor(m.attnOut[i].inputOffset,
                      dev(off, L + "o_proj.input_offset"));
        }
        // quantBias rank0 only (avoid double sum); deqScale not sharded
        // (distributive, same as MLA path).
        if (rank == 0) {
          torch::Tensor qb_o = sd.get_tensor(L + "self_attn.o_proj.quant_bias");
          if (qb_o.defined()) {
            InitXTensor(m.attnOut[i].quantBias,
                        dev(qb_o, L + "o_proj.quant_bias"));
          }
        }
        torch::Tensor ds_o = sd.get_tensor(L + "self_attn.o_proj.deq_scale");
        if (ds_o.defined()) {
          torch::Tensor ds = TransformDeqScale(ds_o);
          InitXTensor(m.attnOut[i].deqScale, dev(ds, L + "o_proj.deq_scale"));
        }
      }

      if (i == 0 && tp > 1) {
        LOG(INFO) << "[xlite] shard layer0 rank=" << rank
                  << ": qkv=" << qkv.sizes() << " o_proj=" << o.sizes();
      }

      if (cfg.addBias) {
        torch::Tensor qb, kb, vb;
        if (tp > 1) {
          qb = Shard(sd, L + "self_attn.q_proj.bias", 0, rank, tp);
          kb = Shard(sd, L + "self_attn.k_proj.bias", 0, kv_rank, kv_tp);
          vb = Shard(sd, L + "self_attn.v_proj.bias", 0, kv_rank, kv_tp);
        } else {
          qb = sd.get_tensor(L + "self_attn.q_proj.bias");
          kb = sd.get_tensor(L + "self_attn.k_proj.bias");
          vb = sd.get_tensor(L + "self_attn.v_proj.bias");
        }
        // bias same as qkv: host cat then one H2D.
        torch::Tensor qkvb =
            torch::cat({qb, kb, vb}, /*dim=*/0).contiguous();  // host
        qkvb = dev(qkvb, L + "self_attn.qkv_bias (mhaQKVBias)");
        InitXTensor(m.mhaQKVBias[i], qkvb);
        storages.push_back(qkvb);
      }
      if (cfg.qkNorm) {
        // q_norm/k_norm: [headDim] shared across heads — no sharding.
        InitXTensor(m.mhaQNorm[i],
                    devBf16(sd.get_tensor(L + "self_attn.q_norm.weight"),
                            L + "self_attn.q_norm.weight"));
        InitXTensor(m.mhaKNorm[i],
                    devBf16(sd.get_tensor(L + "self_attn.k_norm.weight"),
                            L + "self_attn.k_norm.weight"));
        // W8A8 q/k_norm bias; BF16 models have none. Cast to BF16 (same as
        // normBias).
        torch::Tensor qNormBias = sd.get_tensor(L + "self_attn.q_norm.bias");
        if (qNormBias.defined()) {
          torch::Tensor b = qNormBias.to(torch::kBFloat16).contiguous();
          InitXTensor(m.mhaQNormBias[i], dev(b, L + "self_attn.q_norm.bias"));
        }
        torch::Tensor kNormBias = sd.get_tensor(L + "self_attn.k_norm.bias");
        if (kNormBias.defined()) {
          torch::Tensor b = kNormBias.to(torch::kBFloat16).contiguous();
          InitXTensor(m.mhaKNormBias[i], dev(b, L + "self_attn.k_norm.bias"));
        }
      }

      // mlpNorm: not sharded.
      InitXTensor(m.mlpNorm[i],
                  devBf16(sd.get_tensor(L + "post_attention_layernorm.weight"),
                          L + "post_attention_layernorm.weight"));
      // W8A8 mlpNorm bias; BF16 models have none. Cast to BF16 (same as
      // normBias).
      torch::Tensor mlpNormBias =
          sd.get_tensor(L + "post_attention_layernorm.bias");
      if (mlpNormBias.defined()) {
        torch::Tensor b = mlpNormBias.to(torch::kBFloat16).contiguous();
        InitXTensor(m.mlpNormBias[i],
                    dev(b, L + "post_attention_layernorm.bias"));
      }
    }
    LOG(INFO) << "[xlite] LoadAttnAndEmbed: " << cfg.nLayers << " layers done";
  }

  // Dense FFN for layer i (gate_up fuse + down). gate_up dim0, down dim1.
  static void LoadDenseMlp(const StateDict& sd,
                           XModel& m,
                           const XModelConfig& cfg,
                           const ParallelArgs& pa,
                           const torch::Device& device,
                           std::vector<torch::Tensor>& storages,
                           uint32_t i) {
    auto dev = [&](const torch::Tensor& t,
                   const std::string& name) -> const torch::Tensor& {
      return ToDevice(t, name, device, storages);
    };
    const std::string L = "model.layers." + std::to_string(i) + ".";
    const int32_t tp = static_cast<int32_t>(XliteTpSize(pa));
    const int32_t rank = static_cast<int32_t>(XliteTpRank(pa));

    torch::Tensor gp, up, down;
    if (tp > 1) {
      // gate/up: column-parallel dim0. down: row-parallel dim1.
      gp = Shard(sd, L + "mlp.gate_proj.weight", 0, rank, tp);
      up = Shard(sd, L + "mlp.up_proj.weight", 0, rank, tp);
      down = Shard(sd, L + "mlp.down_proj.weight", 1, rank, tp);
    } else {
      gp = sd.get_tensor(L + "mlp.gate_proj.weight");
      up = sd.get_tensor(L + "mlp.up_proj.weight");
      down = sd.get_tensor(L + "mlp.down_proj.weight");
    }
    // Dense MLP: BF16 no transpose (expects [out,in]); W8A8 transpose to
    // [in,out]. mlpDown transpose/NZ see the down_proj block below.
    torch::Tensor gate_up =
        torch::cat({gp, up}, /*dim=*/0).contiguous();  // host [2*inter, hidden]
    if (IsQuant(gp)) {
      gate_up =
          gate_up.t().contiguous();  // W8A8: [hidden, 2*inter] = [in, out]
    }
    gate_up = dev(gate_up, L + "mlp.gate_proj+up_proj (mlpUpGate)");
    // W8A8 INT8 NZ: dense MLP gate_up (quantAttnWeightNz).
    if (IsQuant(gp) && cfg.quantAttnWeightNz) {
      gate_up =
          CastNz(gate_up, L + "mlp.gate_proj+up_proj (mlpUpGate) NZ", storages);
    }
    InitXTensor(m.mlpUpGate[i].weight, gate_up);
    storages.push_back(gate_up);
    // W8A8 quant params: STATIC prefers the deq_scale, DYNAMIC has no
    // deq_scale.
    if (IsQuant(gp)) {
      torch::Tensor gs, us;
      if (tp > 1) {
        gs = Shard(sd, L + "mlp.gate_proj.weight_scale", 0, rank, tp);
        us = Shard(sd, L + "mlp.up_proj.weight_scale", 0, rank, tp);
      } else {
        gs = sd.get_tensor(L + "mlp.gate_proj.weight_scale");
        us = sd.get_tensor(L + "mlp.up_proj.weight_scale");
      }
      // W8A8 STATIC (e.g. Qwen3): deq_scale field present.
      // W8A8 DYNAMIC (e.g. GLM-5.2): deq_scale absent.
      torch::Tensor dsg = sd.get_tensor(L + "mlp.gate_proj.deq_scale");
      if (dsg.defined()) {
        torch::Tensor dsu = sd.get_tensor(L + "mlp.up_proj.deq_scale");
        if (tp > 1) {
          dsg = Shard(sd, L + "mlp.gate_proj.deq_scale", 0, rank, tp);
          dsu = Shard(sd, L + "mlp.up_proj.deq_scale", 0, rank, tp);
        }
        torch::Tensor ds =
            TransformDeqScale(torch::cat({dsg, dsu}, /*dim=*/0).contiguous());
        InitXTensor(m.mlpUpGate[i].deqScale,
                    dev(ds, L + "mlp.gate_proj+up_proj.deq_scale"));
      } else if (gs.defined() && us.defined()) {
        torch::Tensor ds =
            TransformDeqScale(torch::cat({gs, us}, /*dim=*/0).contiguous());
        InitXTensor(m.mlpUpGate[i].deqScale,
                    dev(ds, L + "mlp.gate_proj+up_proj.deq_scale"));
      }
      // W8A8 STATIC (e.g. Qwen3): inputScale/inputOffset (per-tensor, same for
      // gate/up). W8A8 DYNAMIC (e.g. GLM-5.2): no inputScale/inputOffset.
      torch::Tensor iscale = sd.get_tensor(L + "mlp.gate_proj.input_scale");
      torch::Tensor ioffset = sd.get_tensor(L + "mlp.gate_proj.input_offset");
      if (iscale.defined()) {
        torch::Tensor recip = (1.0f / iscale.to(torch::kFloat32))
                                  .to(torch::kBFloat16)
                                  .contiguous();
        recip = recip.repeat({static_cast<int64_t>(cfg.hiddenSize)});
        InitXTensor(m.mlpUpGate[i].inputScale,
                    dev(recip, L + "gate_proj.input_scale_reciprocal"));
      }
      if (ioffset.defined()) {
        torch::Tensor off = ioffset.to(torch::kBFloat16)
                                .contiguous()
                                .repeat({static_cast<int64_t>(cfg.hiddenSize)});
        InitXTensor(m.mlpUpGate[i].inputOffset,
                    dev(off, L + "gate_proj.input_offset"));
      }
      // quantBias: cat(gate,up) + TP shard dim0 (same as weight). STATIC only
      // (e.g. Qwen3); DYNAMIC checkpoints (e.g. GLM-5.2) have none.
      torch::Tensor qb_g = sd.get_tensor(L + "mlp.gate_proj.quant_bias");
      if (qb_g.defined()) {
        torch::Tensor qb_u = sd.get_tensor(L + "mlp.up_proj.quant_bias");
        if (tp > 1) {
          qb_g = Shard(sd, L + "mlp.gate_proj.quant_bias", 0, rank, tp);
          qb_u = Shard(sd, L + "mlp.up_proj.quant_bias", 0, rank, tp);
        }
        torch::Tensor qb = torch::cat({qb_g, qb_u}, /*dim=*/0).contiguous();
        InitXTensor(m.mlpUpGate[i].quantBias,
                    dev(qb, L + "mlp.gate_proj+up_proj.quant_bias"));
      }
    }
    // down_proj: row-parallel (TP shard dim1, ForwardMLP AllReduce).
    // GLM-5.2 down=W8A8_DYNAMIC: transpose + CastNz + deqScale. GLM-4.7
    // down=BF16 (no transpose). row-parallel deqScale not sharded (same as
    // shared expert down).
    torch::Tensor down_w = IsQuant(down) ? down.t().contiguous() : down;
    down_w = dev(down_w, L + "mlp.down_proj.weight");
    if (IsQuant(down) && cfg.quantAttnWeightNz) {
      down_w = CastNz(down_w, L + "mlp.down_proj (mlpDown) NZ", storages);
    }
    InitXTensor(m.mlpDown[i].weight, down_w);
    storages.push_back(down_w);
    if (IsQuant(down)) {
      // W8A8 STATIC (e.g. Qwen3): deq_scale field present.
      // W8A8 DYNAMIC (e.g. GLM-5.2): weight_scale (no deq_scale field).
      torch::Tensor ds_down = sd.get_tensor(L + "mlp.down_proj.deq_scale");
      if (!ds_down.defined()) {
        ds_down = sd.get_tensor(L + "mlp.down_proj.weight_scale");
      }
      if (ds_down.defined()) {
        torch::Tensor ds = TransformDeqScale(ds_down);
        InitXTensor(m.mlpDown[i].deqScale,
                    dev(ds, L + "mlp.down_proj.deq_scale"));
      }
    }
  }

  // MoE FFN for layer i. HF weight [out, in] -> transpose to [in, out]
  // (expertsWeightTrans=true). EP>1: local expert range; EP==1,TP>1: weight
  // split.
  static void LoadMoEExperts(const StateDict& sd,
                             XModel& m,
                             const XModelConfig& cfg,
                             const ParallelArgs& pa,
                             const torch::Device& device,
                             std::vector<torch::Tensor>& storages,
                             uint32_t i) {
    auto dev = [&](const torch::Tensor& t,
                   const std::string& name) -> const torch::Tensor& {
      return ToDevice(t, name, device, storages);
    };
    const std::string L = "model.layers." + std::to_string(i) + ".";

    // moeGate: not sharded (gate logits are all_reduce'd).
    InitXTensor(
        m.moeGate[i],
        dev(sd.get_tensor(L + "mlp.gate.weight"), L + "mlp.gate.weight"));
    // moeGateBias: sigmoid-only (e_score_correction_bias [n_routed_experts]);
    // absent in softmax (Qwen3-MoE).
    if (cfg.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
      torch::Tensor gate_bias =
          sd.get_tensor(L + "mlp.gate.e_score_correction_bias");
      if (gate_bias.defined()) {
        InitXTensor(m.moeGateBias[i],
                    dev(gate_bias, L + "mlp.gate.e_score_correction_bias"));
      }
    }

    // EP: local expert range [start, end).
    const uint32_t ep = cfg.moeEpSize;
    const uint32_t moeTp = cfg.moeTPSize;
    const uint32_t nLocal = cfg.nRoutedExperts / ep;
    const uint32_t rank = static_cast<uint32_t>(pa.rank());
    const uint32_t start = (ep == 1) ? 0 : (rank / moeTp) * nLocal;
    const uint32_t end = start + nLocal;

    for (uint32_t e = start; e < end; ++e) {
      const std::string E = L + "mlp.experts." + std::to_string(e) + ".";
      // tp_rank: EP-group-internal TP rank (rank % moeTp).
      const int32_t tp_rank = static_cast<int32_t>(rank % moeTp);
      torch::Tensor gp, up, down;
      if (moeTp > 1) {
        // TP: gate/up dim0, down dim1.
        gp = Shard(sd,
                   E + "gate_proj.weight",
                   0,
                   tp_rank,
                   static_cast<int32_t>(moeTp));
        up = Shard(
            sd, E + "up_proj.weight", 0, tp_rank, static_cast<int32_t>(moeTp));
        down = Shard(sd,
                     E + "down_proj.weight",
                     1,
                     tp_rank,
                     static_cast<int32_t>(moeTp));
      } else {
        gp = sd.get_tensor(E + "gate_proj.weight");
        up = sd.get_tensor(E + "up_proj.weight");
        down = sd.get_tensor(E + "down_proj.weight");
      }
      // host fused cat+t+contiguous then one H2D.
      torch::Tensor gate_up = torch::cat({gp, up}, /*dim=*/0).t().contiguous();
      gate_up = dev(gate_up, E + "gate_proj+up_proj (moeREUpGate)");
      // W8A8 INT8 NZ: routed expert gate_up (group_matmul).
      if (IsQuant(gp) && cfg.expertsWeightNZ) {
        gate_up =
            CastNz(gate_up, E + "gate_proj+up_proj (moeREUpGate) NZ", storages);
      }
      // Store at original expert index e.
      InitXTensor(m.moeREUpGate[i][e], gate_up);
      storages.push_back(gate_up);
      // down: transpose to [in, out] on host then one H2D.
      torch::Tensor down_t = down.t().contiguous();
      down_t = dev(down_t, E + "down_proj (moeREDown)");
      // W8A8 INT8 NZ: routed expert down (group_matmul).
      if (IsQuant(down) && cfg.expertsWeightNZ) {
        down_t = CastNz(down_t, E + "down_proj (moeREDown) NZ", storages);
      }
      CHECK(down_t.defined())
          << "[xlite] missing weight: " << E + "down_proj.weight";
      InitXTensor(m.moeREDown[i][e], down_t);
      storages.push_back(down_t);
      // W8A8 DYNAMIC: expert gate/up/down weight_scale -> separate deqScale
      // members (group_matmul). gate/up scale shard dim0; down scale
      // row-parallel (not sharded).
      if (IsQuant(gp)) {
        torch::Tensor gs, us;
        if (moeTp > 1) {
          gs = Shard(sd,
                     E + "gate_proj.weight_scale",
                     0,
                     tp_rank,
                     static_cast<int32_t>(moeTp));
          us = Shard(sd,
                     E + "up_proj.weight_scale",
                     0,
                     tp_rank,
                     static_cast<int32_t>(moeTp));
        } else {
          gs = sd.get_tensor(E + "gate_proj.weight_scale");
          us = sd.get_tensor(E + "up_proj.weight_scale");
        }
        if (gs.defined()) {
          torch::Tensor ds =
              TransformDeqScale(torch::cat({gs, us}, /*dim=*/0).contiguous());
          InitXTensor(m.moeREUpGateDeqScale[i][e],
                      dev(ds, E + "gate+up.deq_scale"));
        }
        torch::Tensor ds_down = sd.get_tensor(E + "down_proj.weight_scale");
        if (ds_down.defined()) {
          torch::Tensor ds = TransformDeqScale(ds_down);
          InitXTensor(m.moeREDownDeqScale[i][e], dev(ds, E + "down.deq_scale"));
        }
      }
      if (i == cfg.nDenseLayers && e == start) {
        LOG(INFO) << "[xlite] shard layer" << i << " expert" << e
                  << " rank=" << rank << ": gate_up=" << gate_up.sizes()
                  << " down=" << down_t.sizes();
      }
    }
    // Shared expert (DeepSeek/GLM5, nSharedExperts>0); TP shard by moeTp (same
    // as routed).
    if (cfg.nSharedExperts > 0) {
      const std::string S = L + "mlp.shared_experts.";
      // tp_rank: same as routed expert (EP-group-internal).
      const int32_t tp_rank = static_cast<int32_t>(rank % moeTp);
      torch::Tensor sgp, sup, sdown;
      if (moeTp > 1) {
        sgp = Shard(sd,
                    S + "gate_proj.weight",
                    0,
                    tp_rank,
                    static_cast<int32_t>(moeTp));
        sup = Shard(
            sd, S + "up_proj.weight", 0, tp_rank, static_cast<int32_t>(moeTp));
        sdown = Shard(sd,
                      S + "down_proj.weight",
                      1,
                      tp_rank,
                      static_cast<int32_t>(moeTp));
      } else {
        sgp = sd.get_tensor(S + "gate_proj.weight");
        sup = sd.get_tensor(S + "up_proj.weight");
        sdown = sd.get_tensor(S + "down_proj.weight");
      }
      // Shared expert: BF16 no transpose (ForwardLinear expects [out,in]);
      // W8A8 transpose to [in,out] (same as dense mlpUpGate). host cat then one
      // H2D.

      torch::Tensor sgate_up = torch::cat({sgp, sup}, /*dim=*/0).contiguous();
      bool seQuant = IsQuant(sgp);
      if (seQuant) {
        sgate_up = sgate_up.t().contiguous();
      }
      sgate_up = dev(sgate_up, S + "gate_proj+up_proj (moeSEUpGate)");
      // W8A8 INT8 NZ: shared expert (single expert, not group_matmul).
      if (seQuant && cfg.quantAttnWeightNz) {
        sgate_up = CastNz(
            sgate_up, S + "gate_proj+up_proj (moeSEUpGate) NZ", storages);
      }
      InitXTensor(m.moeSEUpGate[i].weight, sgate_up);
      storages.push_back(sgate_up);
      // shared down: W8A8 transpose to [in,out]; BF16 no transpose.
      torch::Tensor sdown_t = seQuant ? sdown.t().contiguous() : sdown;
      sdown_t = dev(sdown_t, S + "down_proj (moeSEDown)");
      // W8A8 INT8 NZ: shared expert down.
      if (seQuant && cfg.quantAttnWeightNz) {
        sdown_t = CastNz(sdown_t, S + "down_proj (moeSEDown) NZ", storages);
      }
      CHECK(sdown_t.defined())
          << "[xlite] missing weight: " << S + "down_proj.weight";
      InitXTensor(m.moeSEDown[i].weight, sdown_t);
      storages.push_back(sdown_t);
      // W8A8 DYNAMIC: shared expert gate/up/down weight_scale -> ForwardLinear
      // (MatmulWeight.deqScale). gate/up scale shard dim0; down not sharded.
      if (IsQuant(sgp)) {
        torch::Tensor sgs, sus;
        if (moeTp > 1) {
          sgs = Shard(sd,
                      S + "gate_proj.weight_scale",
                      0,
                      tp_rank,
                      static_cast<int32_t>(moeTp));
          sus = Shard(sd,
                      S + "up_proj.weight_scale",
                      0,
                      tp_rank,
                      static_cast<int32_t>(moeTp));
        } else {
          sgs = sd.get_tensor(S + "gate_proj.weight_scale");
          sus = sd.get_tensor(S + "up_proj.weight_scale");
        }
        if (sgs.defined()) {
          torch::Tensor ds =
              TransformDeqScale(torch::cat({sgs, sus}, /*dim=*/0).contiguous());
          InitXTensor(m.moeSEUpGate[i].deqScale,
                      dev(ds, S + "gate+up.deq_scale"));
        }
        torch::Tensor ds_down = sd.get_tensor(S + "down_proj.weight_scale");
        if (ds_down.defined()) {
          torch::Tensor ds = TransformDeqScale(ds_down);
          InitXTensor(m.moeSEDown[i].deqScale, dev(ds, S + "down.deq_scale"));
        }
      }
    }
    if (i == cfg.nDenseLayers) {
      LOG(INFO) << "[xlite] LoadMoEExperts: layer " << i << " loaded "
                << (end - start) << "/" << cfg.nRoutedExperts
                << " experts (ep=" << ep << " moeTp=" << moeTp << " range=["
                << start << "," << end << "))";
    }
  }

  // Dense: attn + dense MLP for all layers.
  static void LoadMHA(const StateDict& sd,
                      XModel& m,
                      const XModelConfig& cfg,
                      const ModelArgs& args,
                      const ParallelArgs& pa,
                      const torch::Device& device,
                      std::vector<torch::Tensor>& storages) {
    LoadAttnAndEmbed(sd, m, cfg, args, pa, device, storages);
    for (uint32_t i = 0; i < cfg.nDenseLayers; ++i) {
      LoadDenseMlp(sd, m, cfg, pa, device, storages, i);
    }
  }

  // FFN binding: dense MLP for i<nDenseLayers, MoE experts for i>=nDenseLayers.
  static void LoadMoEFFN(const StateDict& sd,
                         XModel& m,
                         const XModelConfig& cfg,
                         const ParallelArgs& pa,
                         const torch::Device& device,
                         std::vector<torch::Tensor>& storages) {
    for (uint32_t i = 0; i < cfg.nLayers; ++i) {
      if (i < cfg.nDenseLayers) {
        LoadDenseMlp(sd, m, cfg, pa, device, storages, i);
      } else {
        LoadMoEExperts(sd, m, cfg, pa, device, storages, i);
      }
    }
  }

  // MoE: attn + dense MLP for first nDenseLayers, MoE FFN for the rest.
  static void LoadMoE(const StateDict& sd,
                      XModel& m,
                      const XModelConfig& cfg,
                      const ModelArgs& args,
                      const ParallelArgs& pa,
                      const torch::Device& device,
                      std::vector<torch::Tensor>& storages) {
    LOG(INFO) << "[xlite] LoadMoE: nLayers=" << cfg.nLayers
              << " nDense=" << cfg.nDenseLayers
              << " experts=" << cfg.nRoutedExperts << " act=" << cfg.nActExperts
              << " moeInter=" << cfg.moeIntermediateSize
              << " trans=" << cfg.expertsWeightTrans << " ep=" << cfg.moeEpSize
              << " moeTp=" << cfg.moeTPSize << " rank=" << pa.rank();
    LoadAttnAndEmbed(sd, m, cfg, args, pa, device, storages);
    LoadMoEFFN(sd, m, cfg, pa, device, storages);
    LOG(INFO) << "[xlite] LoadMoE: all layers done, storages="
              << storages.size();
  }

  // MLA attention (DeepSeek-V3/R1, GLM5): bind MLA weights, FFN via LoadMoEFFN.
  static void LoadMLAAttn(const StateDict& sd,
                          XModel& m,
                          const XModelConfig& cfg,
                          const ModelArgs& args,
                          const ParallelArgs& pa,
                          const torch::Device& device,
                          std::vector<torch::Tensor>& storages) {
    auto dev = [&](const torch::Tensor& t,
                   const std::string& name) -> const torch::Tensor& {
      return ToDevice(t, name, device, storages);
    };
    // W8A8 norm weight FP32->BF16 (same as LoadAttnAndEmbed devBf16).
    auto devBf16 = [&](const torch::Tensor& t,
                       const std::string& name) -> const torch::Tensor& {
      if (t.scalar_type() == torch::kBFloat16) {
        return dev(t, name);
      }
      return dev(t.to(torch::kBFloat16).contiguous(), name);
    };
    const int32_t tp = XliteTpSize(pa);
    const int32_t rank = XliteTpRank(pa);

    // embed/head: vocab dim0 (same as MHA); norm not sharded.
    // dev() once and rebind embed so the tie branch can share it (zero-copy).
    torch::Tensor embed =
        (tp > 1) ? Shard(sd, "model.embed_tokens.weight", 0, rank, tp)
                 : sd.get_tensor("model.embed_tokens.weight");
    embed = dev(embed, "model.embed_tokens.weight");
    InitXTensor(m.embed, embed);
    InitXTensor(
        m.norm,
        devBf16(sd.get_tensor("model.norm.weight"), "model.norm.weight"));
    // W8A8 norm bias; BF16 models have none.
    torch::Tensor normBias = sd.get_tensor("model.norm.bias");
    if (normBias.defined()) {
      InitXTensor(m.normBias,
                  devBf16(normBias.contiguous(), "model.norm.bias"));
    }
    torch::Tensor head;
    if (args.tie_word_embeddings()) {
      // tie: reuse embed's device storage (zero-copy). Avoids a second
      // .to(device).
      head = embed;
      InitXTensor(m.head, embed);
    } else {
      head = (tp > 1) ? Shard(sd, "lm_head.weight", 0, rank, tp)
                      : sd.get_tensor("lm_head.weight");
      // GLM-5.2-W8A8: lm_head.weight is FP32; xlite XliteOpMatmul only supports
      // BF16xFP32->FP32. ForwardGetLogits localOutput is BF16 (hiddenState
      // dtype) -> cast to BF16. GLM-5.1 BF16 model: no-op.
      if (head.scalar_type() != torch::kBFloat16) {
        head = head.to(torch::kBFloat16).contiguous();
      }
      InitXTensor(m.head, dev(head, "lm_head.weight"));
    }

    for (uint32_t i = 0; i < cfg.nLayers; ++i) {
      const std::string L = "model.layers." + std::to_string(i) + ".";
      InitXTensor(m.attnNorm[i],
                  devBf16(sd.get_tensor(L + "input_layernorm.weight"),
                          L + "input_layernorm.weight"));
      InitXTensor(m.mlpNorm[i],
                  devBf16(sd.get_tensor(L + "post_attention_layernorm.weight"),
                          L + "post_attention_layernorm.weight"));
      // W8A8 norm biases (attn/mlp); BF16 models have none.
      torch::Tensor attnNormBias = sd.get_tensor(L + "input_layernorm.bias");
      if (attnNormBias.defined()) {
        InitXTensor(
            m.attnNormBias[i],
            devBf16(attnNormBias.contiguous(), L + "input_layernorm.bias"));
      }
      torch::Tensor mlpNormBias =
          sd.get_tensor(L + "post_attention_layernorm.bias");
      if (mlpNormBias.defined()) {
        InitXTensor(m.mlpNormBias[i],
                    devBf16(mlpNormBias.contiguous(),
                            L + "post_attention_layernorm.bias"));
      }

      // mlaQKVA = q_a + kv_a_with_mqa concat (q first; xlite csrc expects
      // q-first). Lora compressed, not sharded.
      torch::Tensor q_a = sd.get_tensor(L + "self_attn.q_a_proj.weight");
      torch::Tensor kv_a =
          sd.get_tensor(L + "self_attn.kv_a_proj_with_mqa.weight");
      CHECK(q_a.defined()) << "[xlite] missing: " << L
                           << "self_attn.q_a_proj.weight";
      CHECK(kv_a.defined())
          << "[xlite] missing: " << L << "self_attn.kv_a_proj_with_mqa.weight";
      // host cat then one H2D. dev(qkva) already pushes the device copy into
      // storages (ToDevice), so no separate push (the old
      // `storages.push_back(qkva)` was a redundant host copy, ~4MB/layer).
      torch::Tensor qkva = torch::cat({q_a, kv_a}, /*dim=*/0)
                               .contiguous();  // [q_lora+kv_lora+rope, hidden]
      // W8A8 (GLM-5.2 STATIC): isTransposed=true, csrc expects [in, out] ->
      // .t(). BF16 no transpose.
      bool qaQuant = IsQuant(q_a);
      if (qaQuant) {
        qkva = qkva.t().contiguous();  // W8A8: [hidden, q_lora+kv_lora+rope] =
                                       // [in, out]
      }
      qkva = dev(qkva, L + "self_attn.mlaQKVA");
      // W8A8 INT8 NZ: quantAttnWeightNz.
      if (qaQuant && cfg.quantAttnWeightNz) {
        qkva = CastNz(qkva, L + "self_attn.mlaQKVA NZ", storages);
      }
      InitXTensor(m.mlaQKVA[i].weight, qkva);
      storages.push_back(qkva);
      // W8A8 STATIC (GLM-5.2): input_scale/input_offset/quant_bias/deq_scale.
      // inputScale: reciprocal + BF16, [1]->[hidden] repeat (per-channel read).
      // Uses q_a's input_scale (q_a/kv_a verified identical).
      if (qaQuant) {
        torch::Tensor iscale =
            sd.get_tensor(L + "self_attn.q_a_proj.input_scale");
        torch::Tensor ioffset =
            sd.get_tensor(L + "self_attn.q_a_proj.input_offset");
        if (iscale.defined()) {
          torch::Tensor recip = (1.0f / iscale.to(torch::kFloat32))
                                    .to(torch::kBFloat16)
                                    .contiguous();
          recip = recip.repeat({static_cast<int64_t>(cfg.hiddenSize)});
          InitXTensor(m.mlaQKVA[i].inputScale,
                      dev(recip, L + "q_a_proj.input_scale_reciprocal"));
        }
        if (ioffset.defined()) {
          torch::Tensor off =
              ioffset.to(torch::kBFloat16)
                  .contiguous()
                  .repeat({static_cast<int64_t>(cfg.hiddenSize)});
          InitXTensor(m.mlaQKVA[i].inputOffset,
                      dev(off, L + "q_a_proj.input_offset"));
        }
        // quantBias/deqScale: cat{q_a, kv_a} not sharded (mlaQKVA
        // column-parallel).
        torch::Tensor qb_qa =
            sd.get_tensor(L + "self_attn.q_a_proj.quant_bias");
        torch::Tensor qb_kva =
            sd.get_tensor(L + "self_attn.kv_a_proj_with_mqa.quant_bias");
        if (qb_qa.defined()) {
          torch::Tensor qb =
              torch::cat({qb_qa, qb_kva}, /*dim=*/0).contiguous();
          InitXTensor(m.mlaQKVA[i].quantBias,
                      dev(qb, L + "mlaQKVA.quant_bias"));
        }
        torch::Tensor ds_qa = sd.get_tensor(L + "self_attn.q_a_proj.deq_scale");
        torch::Tensor ds_kva =
            sd.get_tensor(L + "self_attn.kv_a_proj_with_mqa.deq_scale");
        if (ds_qa.defined()) {
          torch::Tensor ds = TransformDeqScale(
              torch::cat({ds_qa, ds_kva}, /*dim=*/0).contiguous());
          InitXTensor(m.mlaQKVA[i].deqScale, dev(ds, L + "mlaQKVA.deq_scale"));
        }
      }

      // q_a/kv_a_layernorm (RMSNorm, not sharded; BF16 models have no bias).
      InitXTensor(m.mlaQNorm[i],
                  devBf16(sd.get_tensor(L + "self_attn.q_a_layernorm.weight"),
                          L + "self_attn.q_a_layernorm.weight"));
      InitXTensor(m.mlaKVNorm[i],
                  devBf16(sd.get_tensor(L + "self_attn.kv_a_layernorm.weight"),
                          L + "self_attn.kv_a_layernorm.weight"));
      torch::Tensor mlaQNormBias =
          sd.get_tensor(L + "self_attn.q_a_layernorm.bias");
      if (mlaQNormBias.defined()) {
        InitXTensor(m.mlaQNormBias[i],
                    devBf16(mlaQNormBias.contiguous(),
                            L + "self_attn.q_a_layernorm.bias"));
      }
      torch::Tensor mlaKVNormBias =
          sd.get_tensor(L + "self_attn.kv_a_layernorm.bias");
      if (mlaKVNormBias.defined()) {
        InitXTensor(m.mlaKVNormBias[i],
                    devBf16(mlaKVNormBias.contiguous(),
                            L + "self_attn.kv_a_layernorm.bias"));
      }

      // mlaQB = q_b_proj; TP shard dim0 (by n_heads/tp).
      torch::Tensor q_b =
          (tp > 1) ? Shard(sd, L + "self_attn.q_b_proj.weight", 0, rank, tp)
                   : sd.get_tensor(L + "self_attn.q_b_proj.weight");
      // W8A8 (GLM-5.2 STATIC): .t() + CastNz (same as mlaQKVA). BF16 no
      // transpose.
      bool qbQuant = IsQuant(q_b);
      if (qbQuant) {
        q_b =
            q_b.t().contiguous();  // [q_lora, n_heads*(nope+rope)] = [in, out]
      }
      q_b = dev(q_b, L + "self_attn.q_b_proj.weight");
      if (qbQuant && cfg.quantAttnWeightNz) {
        q_b = CastNz(q_b, L + "self_attn.q_b_proj NZ", storages);
      }
      InitXTensor(m.mlaQB[i].weight, q_b);
      storages.push_back(q_b);
      // W8A8 STATIC: mlaQB input_dim=qLoraRank. TP shard dim0;
      // quantBias/deqScale shard dim0.
      if (qbQuant) {
        torch::Tensor iscale =
            sd.get_tensor(L + "self_attn.q_b_proj.input_scale");
        torch::Tensor ioffset =
            sd.get_tensor(L + "self_attn.q_b_proj.input_offset");
        if (iscale.defined()) {
          torch::Tensor recip = (1.0f / iscale.to(torch::kFloat32))
                                    .to(torch::kBFloat16)
                                    .contiguous();
          recip = recip.repeat({static_cast<int64_t>(cfg.qLoraRank)});
          InitXTensor(m.mlaQB[i].inputScale,
                      dev(recip, L + "q_b_proj.input_scale_reciprocal"));
        }
        if (ioffset.defined()) {
          torch::Tensor off =
              ioffset.to(torch::kBFloat16)
                  .contiguous()
                  .repeat({static_cast<int64_t>(cfg.qLoraRank)});
          InitXTensor(m.mlaQB[i].inputOffset,
                      dev(off, L + "q_b_proj.input_offset"));
        }
        torch::Tensor qb_w =
            (tp > 1)
                ? Shard(sd, L + "self_attn.q_b_proj.quant_bias", 0, rank, tp)
                : sd.get_tensor(L + "self_attn.q_b_proj.quant_bias");
        if (qb_w.defined()) {
          InitXTensor(m.mlaQB[i].quantBias,
                      dev(qb_w, L + "q_b_proj.quant_bias"));
        }
        torch::Tensor ds_w =
            (tp > 1)
                ? Shard(sd, L + "self_attn.q_b_proj.deq_scale", 0, rank, tp)
                : sd.get_tensor(L + "self_attn.q_b_proj.deq_scale");
        if (ds_w.defined()) {
          torch::Tensor ds = TransformDeqScale(ds_w);
          InitXTensor(m.mlaQB[i].deqScale, dev(ds, L + "q_b_proj.deq_scale"));
        }
      }

      // kv_b split into WUKT (q absorb) + WUV (output proj); 0.2.0rc0 MLA
      // refactor (old mlaKVB removed). TP shard dim0; host reshape+split then
      // 2x H2D.
      torch::Tensor kv_b =
          (tp > 1) ? Shard(sd, L + "self_attn.kv_b_proj.weight", 0, rank, tp)
                   : sd.get_tensor(L + "self_attn.kv_b_proj.weight");
      CHECK(kv_b.defined())
          << "[xlite] missing: " << L << "self_attn.kv_b_proj.weight";
      // kv_b: [n_local_heads*(nope+v), kv_lora] -> [h, nope+v, kv_lora]
      uint32_t nLocalHeadsMla = cfg.nHeads / tp;
      torch::Tensor wkv_b =
          kv_b.view({static_cast<int64_t>(nLocalHeadsMla),
                     static_cast<int64_t>(cfg.nopeHeadDim + cfg.vHeadDim),
                     static_cast<int64_t>(cfg.kvLoraRank)})
              .contiguous();
      // WUKT: [h, nope, kv_lora], no transpose (matches csrc htd layout).
      torch::Tensor wuk_t = wkv_b.slice(1, 0, cfg.nopeHeadDim).contiguous();
      // WUV: [h, v, kv_lora] -> permute(0,2,1) for csrc EinsumMhtHtdMhd ([h,
      // kv_lora, v]).
      torch::Tensor wuv =
          wkv_b.slice(1, cfg.nopeHeadDim).transpose(1, 2).contiguous();
      InitXTensor(m.mlaWUKT[i], dev(wuk_t, L + "self_attn.kv_b_proj.wuk_t"));
      InitXTensor(m.mlaWUV[i], dev(wuv, L + "self_attn.kv_b_proj.wuv"));

      // attnOut = o_proj; TP shard dim1 (row parallel, AllReduce).
      torch::Tensor o =
          (tp > 1) ? Shard(sd, L + "self_attn.o_proj.weight", 1, rank, tp)
                   : sd.get_tensor(L + "self_attn.o_proj.weight");
      // W8A8 (GLM-5.2 STATIC): .t() + CastNz. row-parallel (shard dim1,
      // AllReduce).
      bool oQuant = IsQuant(o);
      if (oQuant) {
        o = o.t().contiguous();  // [n_heads*v, hidden] = [in, out]
      }
      o = dev(o, L + "self_attn.o_proj.weight");
      if (oQuant && cfg.quantAttnWeightNz) {
        o = CastNz(o, L + "self_attn.o_proj NZ", storages);
      }
      InitXTensor(m.attnOut[i].weight, o);
      storages.push_back(o);
      // W8A8 STATIC: o_proj shard dim1 (AllReduce). quantBias rank0 only (avoid
      // double sum); deqScale not sharded (distributive). inputScale/Offset
      // repeat to nHeads/tp*vHeadDim.
      if (oQuant) {
        uint32_t nLocalHeads = cfg.nHeads / tp;
        int64_t oInputDim = static_cast<int64_t>(nLocalHeads) * cfg.vHeadDim;
        torch::Tensor iscale =
            sd.get_tensor(L + "self_attn.o_proj.input_scale");
        torch::Tensor ioffset =
            sd.get_tensor(L + "self_attn.o_proj.input_offset");
        if (iscale.defined()) {
          torch::Tensor recip = (1.0f / iscale.to(torch::kFloat32))
                                    .to(torch::kBFloat16)
                                    .contiguous();
          recip = recip.repeat({oInputDim});
          InitXTensor(m.attnOut[i].inputScale,
                      dev(recip, L + "o_proj.input_scale_reciprocal"));
        }
        if (ioffset.defined()) {
          torch::Tensor off =
              ioffset.to(torch::kBFloat16).contiguous().repeat({oInputDim});
          InitXTensor(m.attnOut[i].inputOffset,
                      dev(off, L + "o_proj.input_offset"));
        }
        if (rank == 0) {
          torch::Tensor qb_o = sd.get_tensor(L + "self_attn.o_proj.quant_bias");
          if (qb_o.defined()) {
            InitXTensor(m.attnOut[i].quantBias,
                        dev(qb_o, L + "o_proj.quant_bias"));
          }
        }
        torch::Tensor ds_o = sd.get_tensor(L + "self_attn.o_proj.deq_scale");
        if (ds_o.defined()) {
          torch::Tensor ds = TransformDeqScale(ds_o);
          InitXTensor(m.attnOut[i].deqScale, dev(ds, L + "o_proj.deq_scale"));
        }
      }

      // DSA indexer (GLM-5/5.1, attnType==DSA). Weights not TP-sharded (indexer
      // replicated; topkIndices must match across ranks).
      //   indexKWeightsProj = cat{wk, weights_proj} (wk first, no transpose).
      //   indexKNorm/Bias = k_norm (LayerNorm w/ bias). indexQB = wq_b (no
      //   transpose).
      // GLM-5.2 shared layers (cfg.indexFullMask[i]): skip indexer weight
      // binding. Empty list (e.g. consumers without GLM-5.2 support) = full,
      // matching GLM-5.1 behavior.
      if (cfg.attnType == XMODEL_ATTN_DSA &&
          (i >= cfg.indexFullMask.size() || cfg.indexFullMask[i])) {
        torch::Tensor wk = sd.get_tensor(L + "self_attn.indexer.wk.weight");
        torch::Tensor wproj =
            sd.get_tensor(L + "self_attn.indexer.weights_proj.weight");
        CHECK(wk.defined())
            << "[xlite] missing: " << L << "self_attn.indexer.wk.weight";
        CHECK(wproj.defined()) << "[xlite] missing: " << L
                               << "self_attn.indexer.weights_proj.weight";
        wk = dev(wk, L + "self_attn.indexer.wk.weight");
        wproj = dev(wproj, L + "self_attn.indexer.weights_proj.weight");
        torch::Tensor kw_proj = torch::cat({wk, wproj}, /*dim=*/0).contiguous();
        InitXTensor(m.indexKWeightsProj[i], kw_proj);
        storages.push_back(kw_proj);

        InitXTensor(m.indexKNorm[i],
                    dev(sd.get_tensor(L + "self_attn.indexer.k_norm.weight"),
                        L + "self_attn.indexer.k_norm.weight"));
        torch::Tensor indexKNormBias =
            sd.get_tensor(L + "self_attn.indexer.k_norm.bias");
        if (indexKNormBias.defined()) {
          InitXTensor(m.indexKNormBias[i],
                      dev(indexKNormBias.contiguous(),
                          L + "self_attn.indexer.k_norm.bias"));
        }

        torch::Tensor wq_b = sd.get_tensor(L + "self_attn.indexer.wq_b.weight");
        CHECK(wq_b.defined())
            << "[xlite] missing: " << L << "self_attn.indexer.wq_b.weight";
        // W8A8 (GLM-5.2 STATIC): .t() + CastNz. indexer not TP-sharded
        // (replicated).
        bool iqbQuant = IsQuant(wq_b);
        if (iqbQuant) {
          wq_b = wq_b.t().contiguous();  // [in, out]
        }
        wq_b = dev(wq_b, L + "self_attn.indexer.wq_b.weight");
        if (iqbQuant && cfg.quantAttnWeightNz) {
          wq_b = CastNz(wq_b, L + "self_attn.indexer.wq_b NZ", storages);
        }
        InitXTensor(m.indexQB[i].weight, wq_b);
        storages.push_back(wq_b);
        // W8A8 STATIC: indexQB input_dim=indexNHeads*indexHeadDim. indexer
        // replicated (no TP shard).
        if (iqbQuant) {
          int64_t iqInputDim =
              static_cast<int64_t>(cfg.indexNHeads) * cfg.indexHeadDim;
          torch::Tensor iscale =
              sd.get_tensor(L + "self_attn.indexer.wq_b.input_scale");
          torch::Tensor ioffset =
              sd.get_tensor(L + "self_attn.indexer.wq_b.input_offset");
          if (iscale.defined()) {
            torch::Tensor recip = (1.0f / iscale.to(torch::kFloat32))
                                      .to(torch::kBFloat16)
                                      .contiguous();
            recip = recip.repeat({iqInputDim});
            InitXTensor(m.indexQB[i].inputScale,
                        dev(recip, L + "indexer.wq_b.input_scale_reciprocal"));
          }
          if (ioffset.defined()) {
            torch::Tensor off =
                ioffset.to(torch::kBFloat16).contiguous().repeat({iqInputDim});
            InitXTensor(m.indexQB[i].inputOffset,
                        dev(off, L + "indexer.wq_b.input_offset"));
          }
          torch::Tensor qb_iqb =
              sd.get_tensor(L + "self_attn.indexer.wq_b.quant_bias");
          if (qb_iqb.defined()) {
            InitXTensor(m.indexQB[i].quantBias,
                        dev(qb_iqb, L + "indexer.wq_b.quant_bias"));
          }
          torch::Tensor ds_iqb =
              sd.get_tensor(L + "self_attn.indexer.wq_b.deq_scale");
          if (ds_iqb.defined()) {
            torch::Tensor ds = TransformDeqScale(ds_iqb);
            InitXTensor(m.indexQB[i].deqScale,
                        dev(ds, L + "indexer.wq_b.deq_scale"));
          }
        }
      }

      if (i == 0 && tp > 1) {
        LOG(INFO) << "[xlite] MLA shard layer0 rank=" << rank
                  << ": qkva=" << qkva.sizes() << " q_b=" << q_b.sizes()
                  << " o_proj=" << o.sizes();
      }
    }
    LOG(INFO) << "[xlite] LoadMLAAttn: " << cfg.nLayers << " layers done"
              << " (qLora=" << cfg.qLoraRank << " kvLora=" << cfg.kvLoraRank
              << " nope=" << cfg.nopeHeadDim << " rope=" << cfg.ropeHeadDim
              << " v=" << cfg.vHeadDim
              << (cfg.attnType == XMODEL_ATTN_DSA
                      ? " DSA indexNHeads=" + std::to_string(cfg.indexNHeads) +
                            " indexHeadDim=" + std::to_string(cfg.indexHeadDim)
                      : "")
              << ")";
  }

  // DeepSeek-V3/R1 (MLA + MoE): MLA attn + dense FFN + MoE experts.
  static void LoadMLA(const StateDict& sd,
                      XModel& m,
                      const XModelConfig& cfg,
                      const ModelArgs& args,
                      const ParallelArgs& pa,
                      const torch::Device& device,
                      std::vector<torch::Tensor>& storages) {
    LoadMLAAttn(sd, m, cfg, args, pa, device, storages);
    LoadMoEFFN(sd, m, cfg, pa, device, storages);
  }
};

}  // namespace xllm::xlite