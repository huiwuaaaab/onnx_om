from transformers import InternVLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import torch
import inspect

path = "./InternVL3_5-1B-HF"

model = InternVLForConditionalGeneration.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
# with open("internVL_mm_proj.py", "w") as f:
#     f.write(inspect.getsource(model.model.multi_modal_projector.__class__))

image = Image.open('./InternVL3_5-1B-HF/examples/image1.jpg').convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Please describe the image shortly."}
        ]
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = processor(
    text=[prompt],
    images=[image],
    return_tensors='pt'
).to("cuda", torch.bfloat16)


# inputs = processor(
#     text=[prompt],
#     images=[image],
#     return_tensors=None,
#     images_kwargs={"min_patches": 1,
#                 "max_patches": 1,},
# ).to("cuda", torch.bfloat16)

# pixel_values = inputs["pixel_values"]

# if isinstance(pixel_values, list):
#     pixel_values = torch.cat(pixel_values, dim=0)

# inputs["pixel_values"] = pixel_values.unsqueeze(0)

# for k, v in inputs.items():
#     if k != "pixel_values":
#         inputs[k] = torch.tensor(v)

# inputs = {k: v.to("cuda") for k, v in inputs.items()}

# print(inputs['pixel_values'].shape)

outputs = model.generate(**inputs, max_new_tokens=512)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))