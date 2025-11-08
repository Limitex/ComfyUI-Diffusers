import io
import os
from typing import Any

import requests
import torch
from comfy.model_management import get_torch_device  # pyright: ignore[reportMissingImports]
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)
from omegaconf import OmegaConf
from safetensors import safe_open

from ...domain.repositories import AutoencoderRepository
from ...utils import get_cache_path
from ._cache import is_vae_cached


class DiffusersAutoencoderRepository(AutoencoderRepository):
    def __init__(self) -> None:
        self.cache_dir = get_cache_path()
        self.device = get_torch_device()

    # Reference from : https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py
    def convert_and_save_from_single_file(self, checkpoint_path: str) -> str:
        checkpoint_name = os.path.basename(checkpoint_path)
        ckpt_cache_path = os.path.join(self.cache_dir, checkpoint_name)

        # Check if already cached
        if is_vae_cached(ckpt_cache_path):
            return ckpt_cache_path

        self._vae_pt_to_vae_diffuser(checkpoint_path, ckpt_cache_path)
        return ckpt_cache_path

    def load_autoencoder_from_path(self, model_path: str, dtype: torch.dtype) -> AutoencoderKL:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(  # type: ignore[no-untyped-call]
            pretrained_model_name_or_path=model_path,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)
        if vae is None:
            raise RuntimeError(f"Failed to load AutoencoderKL from path: {model_path}")
        return vae

    # --- TECHNICAL DEBT & RISK WARNING ---
    #
    # The following methods (`_custom_convert_ldm_vae_checkpoint` and `_vae_pt_to_vae_diffuser`)
    # are a direct port of the logic from an internal Hugging Face diffusers utility script
    # (scripts/convert_vae_pt_to_diffusers.py).
    #
    # Rationale:
    # This complex, low-level logic is a necessary workaround because the `diffusers`
    # library (as of this writing) does not provide a stable, high-level API
    # to load a VAE from a single checkpoint file (unlike the equivalent
    # `StableDiffusionPipeline.from_single_file`).
    #
    # Risk:
    # This implementation is **EXTREMELY BRITTLE**. It is tightly coupled to the
    # internal key names, architecture, and private utility functions
    # (e.g., `renew_vae_resnet_paths`) of the `diffusers` library.
    #
    # Any future updates to `diffusers` that change the `AutoencoderKL`
    # architecture or refactor these internal utilities will likely **BREAK**
    # this conversion logic silently and catastrophically.
    #
    # TODO:
    # This entire block of code should be aggressively monitored during
    # `diffusers` library upgrades. It should be **IMMEDIATELY DEPRECATED**
    # and replaced if `diffusers` ever releases a stable, official API
    # for this purpose (e.g., `AutoencoderKL.from_single_file(...)`).
    # --- END WARNING ---

    # Reference from : https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py
    def _custom_convert_ldm_vae_checkpoint(
        self, checkpoint: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        vae_state_dict = checkpoint

        new_checkpoint: dict[str, Any] = {}

        new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
        new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
        new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
        new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
        new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
        new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

        new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
        new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
        new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
        new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
        new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
        new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

        new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
        new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
        new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
        new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

        # Retrieves the keys for the encoder down blocks only
        num_down_blocks = len(
            {".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer}
        )
        down_blocks = {
            layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
            for layer_id in range(num_down_blocks)
        }

        # Retrieves the keys for the decoder up blocks only
        num_up_blocks = len(
            {".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer}
        )
        up_blocks = {
            layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
            for layer_id in range(num_up_blocks)
        }

        for i in range(num_down_blocks):
            resnets = [
                key
                for key in down_blocks[i]
                if f"down.{i}" in key and f"down.{i}.downsample" not in key
            ]

            if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
                new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = (
                    vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.weight")
                )
                new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = (
                    vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.bias")
                )

            paths = renew_vae_resnet_paths(resnets)  # type: ignore[no-untyped-call]
            meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
            assign_to_checkpoint(  # type: ignore[no-untyped-call]
                paths,
                new_checkpoint,
                vae_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

        mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
        num_mid_res_blocks = 2
        for i in range(1, num_mid_res_blocks + 1):
            resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

            paths = renew_vae_resnet_paths(resnets)  # type: ignore[no-untyped-call]
            meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
            assign_to_checkpoint(  # type: ignore[no-untyped-call]
                paths,
                new_checkpoint,
                vae_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

        mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
        paths = renew_vae_attention_paths(mid_attentions)  # type: ignore[no-untyped-call]
        meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
        assign_to_checkpoint(  # type: ignore[no-untyped-call]
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )
        conv_attn_to_linear(new_checkpoint)  # type: ignore[no-untyped-call]

        for i in range(num_up_blocks):
            block_id = num_up_blocks - 1 - i
            resnets = [
                key
                for key in up_blocks[block_id]
                if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
            ]

            if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
                new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.weight"
                ]
                new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.bias"
                ]

            paths = renew_vae_resnet_paths(resnets)  # type: ignore[no-untyped-call]
            meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
            assign_to_checkpoint(  # type: ignore[no-untyped-call]
                paths,
                new_checkpoint,
                vae_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

        mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
        num_mid_res_blocks = 2
        for i in range(1, num_mid_res_blocks + 1):
            resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

            paths = renew_vae_resnet_paths(resnets)  # type: ignore[no-untyped-call]
            meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
            assign_to_checkpoint(  # type: ignore[no-untyped-call]
                paths,
                new_checkpoint,
                vae_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

        mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
        paths = renew_vae_attention_paths(mid_attentions)  # type: ignore[no-untyped-call]
        meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
        assign_to_checkpoint(  # type: ignore[no-untyped-call]
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )
        conv_attn_to_linear(new_checkpoint)  # type: ignore[no-untyped-call]
        return new_checkpoint

    # Reference from : https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py
    def _vae_pt_to_vae_diffuser(
        self,
        checkpoint_path: str,
        output_path: str,
    ) -> str:
        # Only support V1
        r = requests.get(
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml",
            timeout=30,
        )
        io_obj = io.BytesIO(r.content)

        original_config = OmegaConf.load(io_obj)
        image_size = 512
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint_path.endswith("safetensors"):
            checkpoint: dict[str, Any] = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call, attr-defined]
                for key in f:
                    checkpoint[key] = f.get_tensor(key)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]

        # Convert the VAE model.
        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
        converted_vae_checkpoint = self._custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)

        vae = AutoencoderKL(**vae_config)  # type: ignore[no-untyped-call]
        vae.load_state_dict(converted_vae_checkpoint)  # type: ignore[attr-defined]
        vae.save_pretrained(output_path)  # type: ignore[attr-defined]

        return output_path
