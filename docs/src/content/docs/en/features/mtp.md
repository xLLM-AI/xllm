---
title: "MTP Speculative Inference"
sidebar:
  order: 80
---
## Background
MTP (Multi-Token Prediction) is an innovative inference acceleration technique that addresses efficiency bottlenecks in large language model generation. By incorporating specialized pre-training designs, MTP provides efficient draft token prediction capabilities during inference, significantly improving generation speed. Its core value lies in balancing inference efficiency with output quality, offering an optimal solution for long-sequence generation problems in LLMs, ultimately optimizing inference performance.

## Key Features
MTP offers the following core acceleration capabilities:

- **Efficient Draft Generation**: Uses a lightweight MTP architecture to rapidly generate draft tokens that serve as input for the main model's verification, dramatically reducing computation overhead compared to traditional autoregressive generation.

- **Batch Verification Mechanism**: The main model can simultaneously verify multiple MTP-generated draft tokens in batch, rather than processing them sequentially, significantly boosting inference speed.

- **High Sampling Accuracy**: MTP solves the critical pain point of low token acceptance rates in post-training draft modules (like Eagle and Medusa). By optimizing draft generation during pre-training, MTP produces tokens with higher accuracy, reducing the verification burden on the main model.

- **Reduced Inference Latency**: By pre-generating multiple potential subsequent tokens, MTP effectively decreases cumulative latency during long-text generation, creating a smoother user experience.

- **Optimized Resource Consumption**: Compared to other inference acceleration techniques, MTP maintains acceleration effects while requiring fewer additional computational resources, making it suitable for deployment in resource-constrained environments.

MTP technology provides a novel efficiency optimization solution for LLM inference, particularly well-suited for real-time applications requiring rapid responses, representing an important direction in language model inference optimization.

:::note[Model Support]
Currently supports MTP structure export for the following models:
- DeepSeek-V3 (input model_type: deepseek_v3, exported MTP model_type: deepseek_v3_mtp)
- DeepSeek-V3.2 (input model_type: deepseek_v3, exported MTP model_type: deepseek_v32_mtp)
- DeepSeek-R1 (input model_type: deepseek_v3, exported MTP model_type: deepseek_v3_mtp)
- GLM4 MoE (e.g., GLM-4.5-Air, exported MTP model_type: glm4_moe_mtp)

Note:
- DeepSeek V3 and R1 both have input model_type "deepseek_v3", and the exported MTP model will have model_type "deepseek_v3_mtp"
- DeepSeek V3.2 has input model_type "deepseek_v3" (but can be auto-detected by index_head_dim fields), and the exported MTP model will have model_type "deepseek_v32_mtp"

:::
## Usage Example

This example assumes the base model is not quantized. For exporting a draft model from a quantized base model, follow this link: [Exporting a draft model from a quantized model](#exporting-a-draft-model-from-a-quantized-model)

### Export Model

The script will automatically detect the model type, or you can manually specify it.

#### DeepSeek-V3
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/DeepSeek-V3 \
    --output-dir /path/to/DeepSeek-V3-mtp
```

#### DeepSeek-V3.2
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/DeepSeek-V3.2 \
    --output-dir /path/to/DeepSeek-V3.2-mtp
```

#### DeepSeek-R1
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/DeepSeek-R1 \
    --output-dir /path/to/DeepSeek-R1-mtp
```

#### GLM4 MoE
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/GLM-4.5-Air \
    --output-dir /path/to/GLM-4.5-Air-mtp
```

#### Manually Specify Model Type
If auto-detection fails, you can manually specify the model type:
```bash
python3 tools/export_mtp.py \
    --input-dir /path/to/model \
    --output-dir /path/to/model-mtp \
    --model-type deepseek_v3  # Options: deepseek_v3 (for V3/R1), deepseek_v32 (for V3.2), glm4_moe
```

Input model references:
- [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air)

### Launch Script

When using MTP for inference, you need to specify both the main model and the draft model (MTP model).

#### DeepSeek-V3/V3.2/R1 Launch Example
```bash
MODEL_PATH="/models/DeepSeek-V3"
DRAFT_MODEL_PATH="/models/DeepSeek-V3-mtp"
MASTER_NODE_ADDR="127.0.0.1:42123"
START_PORT=13222
LOG_DIR="log"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup ./xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --draft_model $DRAFT_MODEL_PATH \
    --num_speculative_tokens 1 \
    --max_memory_utilization=0.90 \
    --max_tokens_per_batch=10000 \
    --max_seqs_per_batch=256 \
    --block_size=128 \
    --ep_size=1 \
    --dp_size=1 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --node_rank=$i > $LOG_FILE 2>&1 &
  sleep 0.5
done
```

#### GLM4 MoE Launch Example
```bash
MODEL_PATH="/models/GLM-4.5-Air"
DRAFT_MODEL_PATH="/models/GLM-4.5-Air-mtp"
# ... same other configurations
```

## Performance Data
Based on ShareGPT dataset with input length=2500, output length=1500, total requests=80.

| method    | Concurrency | Mean TPOT(ms) | Mean TTFT(ms) | Output Tokens/s | Total Tokens/s |
|:---------:|:-----------:|:-------------:|:-------------:|:---------------:|:--------------:|
| baseline  |      1      |     40.61     |    141.80     |      24.20      |     65.77      |
| mtp       |      1      |     28.33     |    142.35     |      35.19      |     95.52      |
| baseline  |      2      |     42.69     |    178.59     |      45.16      |    122.74      |
| mtp       |      2      |     29.81     |    187.97     |      64.75      |    175.78      |
| baseline  |      4      |     46.18     |    172.34     |      79.83      |    216.96      |
| mtp       |      4      |     33.54     |    194.22     |     111.18      |    301.81      |
| baseline  |      8      |     53.16     |    181.49     |     110.68      |    300.81      |
| mtp       |      8      |     40.99     |    203.37     |     154.46      |    419.34      |
| baseline  |     16      |     68.50     |    213.89     |     143.81      |    390.84      |
| mtp       |     16      |     57.04     |    254.99     |     201.89      |    548.04      |
| baseline  |     20      |     74.72     |    228.80     |     154.77      |    420.65      |
| mtp       |     20      |     61.73     |    264.34     |     206.24      |    559.84      |
| baseline  |     40      |    119.68     |    559.32     |     180.22      |    489.80      |
| mtp       |     40      |    105.70     |    544.54     |     252.91      |    686.74      |
| baseline  |     80      |    180.89     |   2996.21     |     192.09      |    522.06      |
| mtp       |     80      |    152.19     |   2163.72     |     278.07      |    755.12      |

## Exporting a draft model from a quantized model

Let's assume we have downloaded a quantized Deepseek-V3-w8a8 model. Unfortunately, if we extract the draft model, this will not be quantized by default; we need to apply quantization once extracted. Here are the steps:

### Export to a temporary draft model

```bash
python3 tools/export_mtp.py --input-dir /path/to/DeepSeek-V3-w8a8 --output-dir /path/to/DeepSeek-V3-w8a8-temp
```

### Patch the temporary draft model's `config.json` file 

- Open the file and change: `"model_type": "deepseek_v3_mtp"` to `"model_type": "deepseek_v3"`.
- Remove the `"quantization_config"` entry.


### Patch the temporary draft model

```bash
cd /path/to/DeepSeek-V3-w8a8-temp

wget https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/configuration_deepseek.py

wget https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/modeling_deepseek.py
```


### Fix broken indexes

```bash
rm -f *.index.json

mv mtp_layer_parameters.safetensors model.safetensors

cat << 'EOF' > /path/to/workspace/make_index.py
import json
from safetensors import safe_open

model_dir = '/path/to/DeepSeek-V3-w8a8-temp'
tensor_file = f'{model_dir}/model.safetensors'

weight_map = {}
# Open the safetensors file and map every tensor inside it to this file
with safe_open(tensor_file, framework="pt", device="cpu") as f:
    for key in f.keys():
        weight_map[key] = "model.safetensors"

# Build the JSON structure both libraries demand
index_data = {
    "metadata": {"total_size": 0},
    "weight_map": weight_map
}

# Save it
with open(f'{model_dir}/model.safetensors.index.json', 'w') as f:
    json.dump(index_data, f, indent=2)

print("Perfect index file created successfully!")
EOF

python3 /path/to/workspace/make_index.py
```

### Install Ascend's ModelSlim toolkit for quantization

```bash
git clone https://gitcode.com/Ascend/msit.git

bash install.sh
```

### Patch ModelSlim

```bash
sed -i 's/patch("transformers.modeling_utils.set_initialized_submodules"/# patch("transformers.modeling_utils.set_initialized_submodules"/g' /path/to/msit/msmodelslim/example/DeepSeek/quant_deepseek_w8a8.py
```

### Generate a quantized draft model from the temporary draft model using ModelSlim

```bash
cd /path/to/msit/msmodelslim/example/DeepSeek

python3 quant_deepseek_w8a8.py --model_path /path/to/DeepSeek-V3-w8a8-temp --save_path /path/to/DeepSeek-V3-w8a8-mtp --batch_size 4 --trust_remote_code True
```

### Patch the quantized draft model's `config.json` file 

- Open the file and change: `"model_type": "deepseek_v3"` to `"model_type": "deepseek_v3_mtp"`.
- Add the entry `"torch_dtype": "bfloat16",` if missing.

### Rescue the quantized draft model weights

```bash
cat << 'EOF' > /path/to/workspace/rescue_mtp.py
import json
import glob
import os
from safetensors import safe_open
from safetensors.torch import save_file

if len(sys.argv) < 3:
    print("Usage: python3 rescue_mtp.py <orig_dir> <quant_dir>")
    sys.exit(1)

orig_dir = sys.argv[1]
quant_dir = sys.argv[2]

# Find original tensor file
orig_tensor_file = f'{orig_dir}/model.safetensors'
if not os.path.exists(orig_tensor_file):
    orig_tensor_file = f'{orig_dir}/mtp_layer_parameters.safetensors'

# Find quantized index file
index_files = glob.glob(f'{quant_dir}/*.index.json')
if not index_files:
    print("Error: Could not find index file in quantized directory.")
    exit(1)
quant_index_file = index_files[0]

# Load original keys
orig_keys = set()
with safe_open(orig_tensor_file, framework="pt", device="cpu") as f:
    orig_keys = set(f.keys())

# Load quantized keys
with open(quant_index_file, 'r') as f:
    quant_index = json.load(f)
quant_keys = set(quant_index['weight_map'].keys())

# Find the ones msmodelslim dropped
missing_keys = orig_keys - quant_keys
print(f"Rescuing {len(missing_keys)} missing weights: {missing_keys}")

if missing_keys:
    # Extract them from the original file
    missing_tensors = {}
    with safe_open(orig_tensor_file, framework="pt", device="cpu") as f:
        for key in missing_keys:
            missing_tensors[key] = f.get_tensor(key)
    
    # Save them into the quantized folder
    out_file = "missing_mtp_weights.safetensors"
    save_file(missing_tensors, f"{quant_dir}/{out_file}")
    
    # Update the JSON map so xllm can find them
    for key in missing_keys:
        quant_index['weight_map'][key] = out_file
        
    with open(quant_index_file, 'w') as f:
        json.dump(quant_index, f, indent=2)
        
    print("Rescue complete! The MTP model is now whole.")
else:
    print("No missing keys found. Something else is wrong.")
EOF

# Run the script
python3 /path/to/workspace/rescue_mtp.py
```

### Delete the temporary draft model

```bash
rm -rf /path/to/DeepSeek-V3-w8a8-temp
```

Now we can go back to starting the server: [Launch Script](#launch-script).
