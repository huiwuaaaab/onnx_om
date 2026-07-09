from transformers import AutoProcessor, AutoModelForCausalLM
import inspect
from PIL import Image
import torch

TARGET_MODEL_ID = "./gemma-4-E2B-it"
ASSISTANT_MODEL_ID = "./gemma-4-E2B-it-assistant"

# Target Model
processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_ID,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"

)
# with open("gemma4it_vision_attn.py", "w") as f:
#     f.write(inspect.getsource(target_model.model.vision_tower.encoder.layers[0].self_attn.__class__))
print(target_model.model.language_model.config._attn_implementation)


# Assistant Model (the drafter)
assistant_model = AutoModelForCausalLM.from_pretrained(
    ASSISTANT_MODEL_ID,
    dtype="auto",
    device_map="auto",
)

image = Image.open('../../imgs/example.jpg').convert("RGB").resize((768, 768))
# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True
).to(target_model.device)
input_len = inputs["input_ids"].shape[-1]

#input_ids:[5:261]为image token

# =====  PATCH 坐标 & PADDING 位置详情  =====
# pixel_values shape: torch.Size([1, 2520, 768])
# image_position_ids shape: torch.Size([1, 2520, 2])

# [有效 patch 部分]
#   起始索引: 0
#   结束索引: 2303
#   总数: 2304

# [Padding 部分]
#   起始索引: 2304
#   结束索引: 2519
#   总数: 216

# Generate output
outputs = target_model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)

print(response)
