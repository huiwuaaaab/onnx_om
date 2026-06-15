import time
import copy
import numpy as np
import torch
import onnxruntime as ort

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm

# =========================================================
# Config
# =========================================================
MODEL_PATH = "./gemma-4-E2B-it"

USE_ONNX_CUDA = False
DEVICE = "cpu"

MAX_LEN = 512

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PRINT_SHAPES = True


def _array_info(arr) -> str:
    if isinstance(arr, np.ndarray):
        return f"shape={arr.shape}, dtype={arr.dtype}"
    if isinstance(arr, torch.Tensor):
        return f"shape={tuple(arr.shape)}, dtype={arr.dtype}, device={arr.device}"
    return f"type={type(arr).__name__}"


def log_tensors(title: str, **tensors) -> None:
    if not PRINT_SHAPES:
        return
    print(f"\n--- {title} ---")
    for name, val in tensors.items():
        print(f"  {name:26s} {_array_info(val)}")


def log_onnx_session(sess: ort.InferenceSession, name: str) -> None:
    if not PRINT_SHAPES:
        return
    print(f"\n=== ONNX {name} ===")
    for inp in sess.get_inputs():
        print(f"  in  {inp.name:22s} {inp.type:18s} {inp.shape}")
    for out in sess.get_outputs():
        print(f"  out {out.name:22s} {out.type:18s} {out.shape}")


def log_onnx_inputs_dict(onnx_inputs: dict) -> None:
    log_tensors(
        "ONNX inputs",
        pixel_values=onnx_inputs["pixel_values"],
        image_position_ids=onnx_inputs["image_position_ids"],
        input_ids=onnx_inputs["input_ids"],
        attention_mask=onnx_inputs["attention_mask"],
        per_layer_inputs=onnx_inputs["per_layer_inputs"],
        attn_sum=np.array([onnx_inputs["attention_mask"].sum()]),
    )


def log_llm_split_outputs(hidden, logits, full_k, full_v, slide_k, slide_v) -> None:
    log_tensors(
        "LLM split outputs",
        hidden=hidden,
        logits=logits,
        full_k=full_k,
        full_v=full_v,
        slide_k=slide_k,
        slide_v=slide_v,
    )

# =========================================================
# Load ONNX
# =========================================================
def load_onnx():

    providers = (
        ["CUDAExecutionProvider"]
        if USE_ONNX_CUDA
        else ["CPUExecutionProvider"]
    )

    vision = ort.InferenceSession(
        "./onnx_export/vision.onnx",
        providers=providers
    )

    mm_proj = ort.InferenceSession(
        "./onnx_export/mm_proj.onnx",
        providers=providers
    )

    ort_pre = ort.InferenceSession(
        "./onnx_export/llm_preblock.onnx",
        providers=providers
    )

    ort_b1 = ort.InferenceSession(
        "./onnx_export/llm_block_0_5.onnx",
        providers=providers
    )

    ort_b2 = ort.InferenceSession(
        "./onnx_export/llm_block_5_10.onnx",
        providers=providers
    )

    ort_b3 = ort.InferenceSession(
        "./onnx_export/llm_block_10_15.onnx",
        providers=providers
    )

    ort_b4 = ort.InferenceSession(
        "./onnx_export/llm_block_15_20.onnx",
        providers=providers
    )

    ort_b5 = ort.InferenceSession(
        "./onnx_export/llm_block_20_25.onnx",
        providers=providers
    )

    ort_b6 = ort.InferenceSession(
        "./onnx_export/llm_block_25_30.onnx",
        providers=providers
    )

    ort_b7 = ort.InferenceSession(
        "./onnx_export/llm_block_30_35.onnx",
        providers=providers
    )

    ort_assistant = ort.InferenceSession(
        "./onnx_export/assistant.onnx",
        providers=providers
    )

    print(f"🚀 ONNX Provider: {providers}")

    sessions = (
        vision,
        mm_proj,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_assistant,
    )
    names = (
        "vision",
        "mm_proj",
        "llm_preblock",
        "llm_block_0_5",
        "llm_block_5_10",
        "llm_block_10_15",
        "llm_block_15_20",
        "llm_block_20_25",
        "llm_block_25_30",
        "llm_block_30_35",
        "assistant",
    )
    for sess, name in zip(sessions, names):
        log_onnx_session(sess, name)

    return sessions

# =========================================================
# Preprocess
# =========================================================
def preprocess(processor):

    image = Image.open(
        "path/to/image.jpg"
    ).convert("RGB").resize((768, 768))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is shown in this image?"}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE).to(torch.float16)

    return inputs

# =========================================================
# Build ONNX Inputs
# =========================================================
@torch.no_grad()
def build_onnx_inputs(model, inputs):

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    pixel_values = inputs["pixel_values"]
    image_position_ids = inputs["image_position_ids"].to(torch.int32)

    B, L = input_ids.shape

    pad_len = MAX_LEN - L
    pad_id = 0

    # =====================================================
    # pad input ids
    # =====================================================
    pad_ids = torch.full(
        (B, pad_len),
        pad_id,
        dtype=input_ids.dtype,
        device=input_ids.device
    )

    pad_mask = torch.zeros(
        (B, pad_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device
    )

    input_ids_pad = torch.cat(
        [input_ids, pad_ids],
        dim=1
    )

    attention_mask_pad = torch.cat(
        [attention_mask, pad_mask],
        dim=1
    )

    # =====================================================
    # placeholder mask
    # =====================================================
    llm_input_ids = input_ids_pad.clone()

    image_mask, _, _ = model.model.get_placeholder_mask(
        llm_input_ids
    )

    llm_input_ids[image_mask] = (
        model.model.language_model.config.pad_token_id
    )

    # =====================================================
    # token embeddings
    # =====================================================
    inputs_embeds = model.model.get_input_embeddings()(
        llm_input_ids
    )

    # =====================================================
    # pad embedding
    # =====================================================
    pad_embedding = (
        model.model.language_model.embed_tokens.weight[
            model.model.config.text_config.pad_token_id
        ]
    )

    image_mask = image_mask.to(inputs_embeds.device)

    llm_inputs_embeds = torch.where(
        image_mask[..., None],
        pad_embedding.view(1, 1, -1),
        inputs_embeds
    )

    # =====================================================
    # per layer inputs
    # =====================================================
    per_layer_inputs = (
        model.model.language_model.get_per_layer_inputs(
            llm_input_ids,
            llm_inputs_embeds
        )
    )

    return {
        "pixel_values":
            pixel_values.cpu().numpy(),

        "image_position_ids":
            image_position_ids.cpu().numpy(),

        "input_ids":
            input_ids_pad.to(torch.int32).cpu().numpy(),

        "attention_mask":
            attention_mask_pad.to(torch.int32).cpu().numpy(),

        "per_layer_inputs":
            per_layer_inputs.cpu().numpy().astype(np.float16),
    }

# =========================================================
# dtype helpers (preblock fp16 <-> block fp16/fp32)
# =========================================================
def _onnx_float_dtype(session: ort.InferenceSession) -> np.dtype:
    elem_type = session.get_inputs()[0].type
    return np.float16 if "float16" in elem_type else np.float32


def _cast_float_array(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype in (np.float16, np.float32, np.float64):
        return arr.astype(dtype, copy=False)
    return arr

# =========================================================
# Vision Encoder
# =========================================================
def run_onnx_vision(
    vision_sess,
    pixel_values,
    image_position_ids
):

    image_embeds = vision_sess.run(
        None,
        {
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids
        }
    )[0]

    return image_embeds

# =========================================================
# MM Projector
# =========================================================
def run_onnx_mmproj(
    proj_sess,
    image_embeds
):

    proj_embeds = proj_sess.run(
        None,
        {
            "vision_features": image_embeds
        }
    )[0]

    return proj_embeds

# =========================================================
# Main LLM
# =========================================================
def run_onnx_llm_split(
    ort_pre,
    ort_b1,
    ort_b2,
    ort_b3,
    ort_b4,
    ort_b5,
    ort_b6,
    ort_b7,
    onnx_inputs,
    image_embeds
):

    block_dtype = _onnx_float_dtype(ort_b1)
    if PRINT_SHAPES and not getattr(run_onnx_llm_split, "_logged_block_dtype", False):
        print(f"\n--- LLM block dtype (from ort_b1) ---\n  block_dtype = {block_dtype}")
        run_onnx_llm_split._logged_block_dtype = True

    # =====================================================
    # pre block
    # =====================================================
    pre_inputs = {
        "input_ids":
            onnx_inputs["input_ids"],

        "image_embeds":
            image_embeds,

        "attention_mask":
            onnx_inputs["attention_mask"],

        "per_layer_inputs":
            onnx_inputs["per_layer_inputs"],
    }

    (
        inputs_embeds,
        per_layer_inputs,
        full_mask,
        sliding_mask,
        cos_full,
        sin_full,
        cos_slide,
        sin_slide
    ) = ort_pre.run(None, pre_inputs)

    inputs_embeds = _cast_float_array(inputs_embeds, block_dtype)
    per_layer_inputs = _cast_float_array(per_layer_inputs, block_dtype)
    full_mask = _cast_float_array(full_mask, block_dtype)
    sliding_mask = _cast_float_array(sliding_mask, block_dtype)
    cos_full = _cast_float_array(cos_full, block_dtype)
    sin_full = _cast_float_array(sin_full, block_dtype)
    cos_slide = _cast_float_array(cos_slide, block_dtype)
    sin_slide = _cast_float_array(sin_slide, block_dtype)

    # =====================================================
    # block 1
    # =====================================================
    hidden = ort_b1.run(
        None,
        {
            "inputs_embeds": inputs_embeds,
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
            "per_layer_input":
                per_layer_inputs[:, :, 0:5, :]
        }
    )[0]

    # =====================================================
    # block 2
    # =====================================================
    hidden = ort_b2.run(
        None,
        {
            "inputs_embeds": hidden,
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
            "per_layer_input":
                per_layer_inputs[:, :, 5:10, :]
        }
    )[0]

    # =====================================================
    # block 3
    # =====================================================
    hidden, out_full_k, out_full_v, out_slide_k, out_slide_v = ort_b3.run(
        None,
        {
            "inputs_embeds": hidden,
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
            "per_layer_input":
                per_layer_inputs[:, :, 10:15, :]
        }
    )

    full_k = out_full_k
    full_v = out_full_v

    slide_k = out_slide_k
    slide_v = out_slide_v

    # =====================================================
    # block 4
    # =====================================================
    hidden = ort_b4.run(
        None,
        {
            "inputs_embeds": hidden,
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
            "full_k": full_k,
            "full_v": full_v,
            "slide_k": slide_k,
            "slide_v": slide_v,
            "per_layer_input":
                per_layer_inputs[:, :, 15:20, :]
        }
    )[0]

    # =====================================================
    # block 5
    # =====================================================
    hidden = ort_b5.run(
        None,
        {
            "inputs_embeds": hidden,
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
            "full_k": full_k,
            "full_v": full_v,
            "slide_k": slide_k,
            "slide_v": slide_v,
            "per_layer_input":
                per_layer_inputs[:, :, 20:25, :]
        }
    )[0]

    # =====================================================
    # block 6
    # =====================================================
    hidden = ort_b6.run(
        None,
        {
            "inputs_embeds": hidden,
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
            "full_k": full_k,
            "full_v": full_v,
            "slide_k": slide_k,
            "slide_v": slide_v,
            "per_layer_input":
                per_layer_inputs[:, :, 25:30, :]
        }
    )[0]

    # =====================================================
    # block 7
    # =====================================================
    hidden, logits = ort_b7.run(
        None,
        {
            "inputs_embeds": hidden,
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
            "full_k": full_k,
            "full_v": full_v,
            "slide_k": slide_k,
            "slide_v": slide_v,
            "per_layer_input":
                per_layer_inputs[:, :, 30:35, :]
        }
    )

    return (
        hidden,
        logits,
        full_k,
        full_v,
        slide_k,
        slide_v
    )

# =========================================================
# Assistant
# =========================================================
def run_assistant_split(
    ort_assistant,
    last_hidden,
    last_token_id,
    position_ids,
    attention_mask,
    full_k,
    full_v,
    slide_k,
    slide_v
):

    assistant_inputs = {
        "last_token_id": last_token_id,
        "last_hidden": last_hidden,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "full_k": full_k,
        "full_v": full_v,
        "slide_k": slide_k,
        "slide_v": slide_v
    }

    projected_state, logits, _ = ort_assistant.run(
        None,
        assistant_inputs
    )

    return projected_state, logits


def _get_eos_token_ids(model) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.model.config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.model.config.text_config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple)):
        return {int(x) for x in eos}
    return {int(eos)}


def _token_is_eos(token, eos_ids: set[int]) -> bool:
    if not eos_ids:
        return False
    return int(np.asarray(token).reshape(-1)[0]) in eos_ids


@torch.no_grad()
def recompute_per_layer_inputs(model, onnx_inputs):
    """按完整 input_ids 重算 PLE（校验草稿序列前必须调用）。"""
    device = model.device
    input_ids_pt = torch.from_numpy(onnx_inputs["input_ids"]).to(device=device, dtype=torch.long)
    image_mask, _, _ = model.model.get_placeholder_mask(input_ids_pt)
    pad_id = model.model.language_model.config.pad_token_id
    input_ids_pt = input_ids_pt.clone()
    input_ids_pt[image_mask] = pad_id

    embeds = model.model.get_input_embeddings()(input_ids_pt)
    pad_embedding = model.model.language_model.embed_tokens.weight[
        model.model.config.text_config.pad_token_id
    ]
    llm_embeds = torch.where(image_mask[..., None], pad_embedding.view(1, 1, -1), embeds)
    ple = model.model.language_model.get_per_layer_inputs(input_ids_pt, llm_embeds)
    onnx_inputs["per_layer_inputs"] = ple.cpu().numpy().astype(np.float16)


@torch.no_grad()
def update_per_layer_inputs(
    model,
    onnx_inputs,
    start_pos,
    new_token_ids
):

    device = model.device

    # ============================================
    # numpy -> torch
    # ============================================
    new_token_ids_pt = torch.from_numpy(
        new_token_ids
    ).to(device=device, dtype=torch.long)

    # ============================================
    # image placeholder mask
    # ============================================
    image_mask, _, _ = model.model.get_placeholder_mask(
        new_token_ids_pt
    )

    new_token_ids_pt = new_token_ids_pt.clone()

    new_token_ids_pt[image_mask] = (
        model.model.language_model.config.pad_token_id
    )

    # ============================================
    # token embeddings
    # ============================================
    embeds = model.model.get_input_embeddings()(
        new_token_ids_pt
    )

    # ============================================
    # pad embedding
    # ============================================
    pad_embedding = (
        model.model.language_model.embed_tokens.weight[
            model.model.config.text_config.pad_token_id
        ]
    )

    llm_embeds = torch.where(
        image_mask[..., None],
        pad_embedding.view(1, 1, -1),
        embeds
    )

    # ============================================
    # NEW per layer inputs
    # ============================================
    new_per_layer_inputs = (
        model.model.language_model.get_per_layer_inputs(
            new_token_ids_pt,
            llm_embeds
        )
    ).cpu().numpy().astype(np.float16)

    # ============================================
    # inplace update
    # ============================================
    N = new_token_ids.shape[1]

    onnx_inputs["per_layer_inputs"][
        :,
        start_pos:start_pos + N,
        :,
        :
    ] = new_per_layer_inputs

# =========================================================
# ONNX Generate
# =========================================================
@torch.no_grad()
def generate_onnx(
    model,
    ort_vision,
    ort_proj,
    ort_pre,
    ort_b1,
    ort_b2,
    ort_b3,
    ort_b4,
    ort_b5,
    ort_b6,
    ort_b7,
    ort_assistant,
    onnx_inputs,
    steps=100,
    num_assistant_tokens=10
):

    # =====================================================
    # vision
    # =====================================================
    image_embeds = run_onnx_vision(
        ort_vision,
        onnx_inputs["pixel_values"],
        onnx_inputs["image_position_ids"]
    )

    vision_out = image_embeds
    image_embeds = run_onnx_mmproj(
        ort_proj,
        image_embeds
    )

    log_tensors(
        "Vision path",
        vision_pooler=vision_out,
        mm_proj_out=image_embeds,
    )

    # =====================================================
    # first forward
    # =====================================================
    hidden, logits, full_k, full_v, slide_k, slide_v = (
        run_onnx_llm_split(
            ort_pre,
            ort_b1,
            ort_b2,
            ort_b3,
            ort_b4,
            ort_b5,
            ort_b6,
            ort_b7,
            onnx_inputs,
            image_embeds
        )
    )

    cur_len = int(onnx_inputs["attention_mask"].sum())

    log_llm_split_outputs(hidden, logits, full_k, full_v, slide_k, slide_v)
    log_tensors("Prefill", cur_len=np.array([cur_len]))

    generated_tokens = []

    accept_history = []

    total_draft_tokens = 0
    total_accept_tokens = 0

    eos_ids = _get_eos_token_ids(model)
    stop_generation = False

    # =====================================================
    # generation loop
    # =====================================================
    for step in tqdm(range(steps)):

        if cur_len >= MAX_LEN or stop_generation:
            break

        # =================================================
        # assistant draft
        # =================================================
        candidate_tokens = []

        last_hidden = hidden[:, cur_len - 1:cur_len]

        last_token_id = (
            onnx_inputs["input_ids"][
                :,
                cur_len - 1:cur_len
            ]
        )

        attention_mask = (
            onnx_inputs["attention_mask"]
        )

        ass_fk = full_k.copy()
        ass_fv = full_v.copy()

        ass_sk = slide_k.copy()
        ass_sv = slide_v.copy()

        for k_idx in range(num_assistant_tokens):

            pos = cur_len + k_idx

            if pos >= MAX_LEN:
                break

            position_ids = np.array(
                [[cur_len - 1]],
                dtype=np.int32
            )

            if step == 0 and k_idx == 0:
                log_tensors(
                    "Assistant inputs (step 0)",
                    last_hidden=last_hidden,
                    last_token_id=last_token_id,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    full_k=ass_fk,
                    full_v=ass_fv,
                    slide_k=ass_sk,
                    slide_v=ass_sv,
                )

            ass_hidden, ass_logits = run_assistant_split(
                ort_assistant,
                last_hidden,
                last_token_id,
                position_ids,
                attention_mask,
                ass_fk,
                ass_fv,
                ass_sk,
                ass_sv
            )

            pred_token = np.argmax(
                ass_logits,
                axis=-1
            )

            candidate_tokens.append(pred_token)

            last_hidden = ass_hidden
            last_token_id = pred_token.astype(np.int32)

            if _token_is_eos(pred_token, eos_ids):
                break

        if not candidate_tokens:
            break

        cand_tokens = np.concatenate(
            candidate_tokens,
            axis=1
        )

        cand_len = cand_tokens.shape[1]

        # =================================================
        # verify
        # =================================================
        temp_ids = (
            onnx_inputs["input_ids"].copy()
        )

        temp_mask = (
            onnx_inputs["attention_mask"].copy()
        )

        temp_ids[
            :,
            cur_len:cur_len + cand_len
        ] = cand_tokens

        temp_mask[
            :,
            cur_len:cur_len + cand_len
        ] = 1

        temp_inputs = copy.deepcopy(onnx_inputs)

        temp_inputs["input_ids"] = temp_ids
        temp_inputs["attention_mask"] = temp_mask
        recompute_per_layer_inputs(model, temp_inputs)

        verify_hidden, verify_logits, v_fk, v_fv, v_sk, v_sv = (
            run_onnx_llm_split(
                ort_pre,
                ort_b1,
                ort_b2,
                ort_b3,
                ort_b4,
                ort_b5,
                ort_b6,
                ort_b7,
                temp_inputs,
                image_embeds
            )
        )

        if step == 0:
            log_tensors(
                "Verify inputs (step 0)",
                temp_input_ids=temp_ids,
                temp_attention_mask=temp_mask,
                temp_per_layer_inputs=temp_inputs["per_layer_inputs"],
                cand_tokens=cand_tokens,
            )
            log_llm_split_outputs(
                verify_hidden, verify_logits, v_fk, v_fv, v_sk, v_sv
            )

        main_pred = np.argmax(
            verify_logits[
                :,
                cur_len - 1:cur_len - 1 + cand_len +1,
                :
            ],
            axis=-1
        )

        # =================================================
        # match count
        # =================================================
        match_cnt = 0

        for i in range(cand_len):

            if cand_tokens[0, i] == main_pred[0, i]:
                match_cnt += 1
            else:
                break

        # accept matched + one extra
        match_cnt = 0

        for i in range(cand_len):

            if cand_tokens[0, i] == main_pred[0, i]:
                match_cnt += 1
            else:
                break

        # ==========================================
        # assistant accepted tokens
        # ==========================================
        accepted_assistant_tokens = match_cnt

        # ==========================================
        # actual appended tokens
        # matched assistant tokens
        # + one extra main-model token
        # ==========================================
        accept_tokens = main_pred[:, :match_cnt + 1]

        if step == 0:
            log_tensors(
                "Verify preds (step 0)",
                main_pred=main_pred,
                accept_tokens=accept_tokens,
                match_cnt=np.array([match_cnt]),
            )

        accept_num = match_cnt + 1

        # ==========================================
        # statistics
        # ==========================================
        accept_history.append(
            accepted_assistant_tokens
        )

        total_accept_tokens += (
            accepted_assistant_tokens
        )

        total_draft_tokens += cand_len

        generated_tokens.append(accept_tokens)

        # =================================================
        # update global
        # =================================================
        onnx_inputs["input_ids"][
            :,
            cur_len:cur_len + accept_num
        ] = accept_tokens

        onnx_inputs["attention_mask"][
            :,
            cur_len:cur_len + accept_num
        ] = 1

        update_per_layer_inputs(
            model,
            onnx_inputs,
            cur_len,
            accept_tokens
        )

        cur_len += accept_num

        hidden = verify_hidden

        full_k = v_fk
        full_v = v_fv

        slide_k = v_sk
        slide_v = v_sv

        if eos_ids:
            for i in range(accept_num):
                if int(accept_tokens[0, i]) in eos_ids:
                    stop_generation = True
                    break

    # =====================================================
    # concat outputs
    # =====================================================
    accept_arr = np.array(accept_history)

    mean_accept = accept_arr.mean()
    var_accept = accept_arr.var()
    std_accept = accept_arr.std()

    max_accept = accept_arr.max()
    min_accept = accept_arr.min()

    accept_rate = (
        total_accept_tokens / total_draft_tokens
        if total_draft_tokens > 0 else 0
    )

    print("\n==============================")
    print("📊 Speculative Decoding Stats")
    print("==============================")

    print(f"Steps                  : {len(accept_history)}")
    print(f"Mean Accept Tokens     : {mean_accept:.4f}")
    print(f"Variance               : {var_accept:.4f}")
    print(f"Std                    : {std_accept:.4f}")

    print(f"Max Accept             : {max_accept}")
    print(f"Min Accept             : {min_accept}")

    print(f"Total Draft Tokens     : {total_draft_tokens}")
    print(f"Total Accepted Tokens  : {total_accept_tokens}")

    print(f"Acceptance Rate        : {accept_rate:.4f}")

    print("==============================\n")

    all_tokens = np.concatenate(
        generated_tokens,
        axis=1
    )

    return all_tokens

# =========================================================
# Main
# =========================================================
def main():

    # =====================================================
    # torch model
    # only for per_layer_inputs construction
    # =====================================================
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    ).to(DEVICE).to(torch.float16).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH
    )

    # =====================================================
    # ONNX
    # =====================================================
    (
        ort_vision,
        ort_proj,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_assistant
    ) = load_onnx()

    # =====================================================
    # inputs
    # =====================================================
    inputs = preprocess(processor)

    log_tensors(
        "HF inputs (processor)",
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_position_ids=inputs["image_position_ids"],
    )

    onnx_inputs = build_onnx_inputs(
        model,
        inputs
    )

    log_onnx_inputs_dict(onnx_inputs)

    # =====================================================
    # generate
    # =====================================================
    start = time.time()

    out_ids = out_ids = generate_onnx(
        model,
        ort_vision,
        ort_proj,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_assistant,
        onnx_inputs,
        steps=100,
        num_assistant_tokens=10
    )[0]

    end = time.time()

    # =====================================================
    # decode
    # =====================================================
    text = processor.decode(
        out_ids,
        skip_special_tokens=False
    )

    print("\n==============================")
    print("🤖 ONNX OUTPUT")
    print("==============================")
    print(text)

    print("\n==============================")
    print(f"⏱ Time: {end - start:.2f}s")
    print("==============================")

if __name__ == "__main__":
    main()