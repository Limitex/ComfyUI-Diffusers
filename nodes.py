import copy
import os
import gc
import torch
import traceback
from safetensors.torch import load_file
from .utils import SCHEDULERS, token_auto_concat_embeds, vae_pt_to_vae_diffuser, convert_images_to_tensors, convert_tensors_to_images, resize_images
from comfy.model_management import get_torch_device
import folder_paths
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoencoderTiny
from pathlib import Path

class DiffusersPipelineLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ), }}

    RETURN_TYPES = ("PIPELINE", "AUTOENCODER", "SCHEDULER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, ckpt_name):
        ckpt_cache_path = os.path.join(self.tmp_dir, ckpt_name)
        
        StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=folder_paths.get_full_path("checkpoints", ckpt_name),
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        ).save_pretrained(ckpt_cache_path, safe_serialization=True)
        
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt_cache_path,
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        )
        return ((pipe, ckpt_cache_path), pipe.vae, pipe.scheduler)

class DiffusersVaeLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (folder_paths.get_filename_list("vae"), ), }}

    RETURN_TYPES = ("AUTOENCODER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, vae_name):
        ckpt_cache_path = os.path.join(self.tmp_dir, vae_name)
        vae_pt_to_vae_diffuser(folder_paths.get_full_path("vae", vae_name), ckpt_cache_path)

        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=ckpt_cache_path,
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        )
        
        return (vae,)

class DiffusersSchedulerLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", ),
                "scheduler_name": (list(SCHEDULERS.keys()), ), 
            }
        }

    RETURN_TYPES = ("SCHEDULER",)

    FUNCTION = "load_scheduler"

    CATEGORY = "Diffusers"

    def load_scheduler(self, pipeline, scheduler_name):
        scheduler = SCHEDULERS[scheduler_name].from_pretrained(
            pretrained_model_name_or_path=pipeline[1],
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
            subfolder='scheduler'
        )
        return (scheduler,)

class DiffusersModelMakeup:
    def __init__(self):
        self.torch_device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", ), 
                "scheduler": ("SCHEDULER", ),
                "autoencoder": ("AUTOENCODER", ),
            }, 
        }

    RETURN_TYPES = ("MAKED_PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "Diffusers"

    def makeup_pipeline(self, pipeline, scheduler, autoencoder):
        pipe = pipeline[0]
        ckpt_cache_path = pipeline[1]
        pipe.vae = autoencoder
        pipe.scheduler = scheduler
        pipe.safety_checker = None if pipe.safety_checker is None else lambda images, **kwargs: (images, [False])
        pipe.enable_attention_slicing()
        pipe = pipe.to(self.torch_device)
        return ((pipe, ckpt_cache_path),)

class DiffusersClipTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "maked_pipeline": ("MAKED_PIPELINE", ),
            "positive": ("STRING", {"multiline": True}),
            "negative": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("EMBEDS", "EMBEDS", "STRING", "STRING", )
    RETURN_NAMES = ("positive_embeds", "negative_embeds", "positive", "negative", )

    FUNCTION = "concat_embeds"

    CATEGORY = "Diffusers"

    def concat_embeds(self, maked_pipeline, positive, negative):
        positive_embeds, negative_embeds = token_auto_concat_embeds(maked_pipeline[0], positive,negative)

        return (positive_embeds, negative_embeds, positive, negative, )

class DiffusersSampler:
    def __init__(self):
        self.torch_device = get_torch_device()
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "maked_pipeline": ("MAKED_PIPELINE", ),
            "positive_embeds": ("EMBEDS", ),
            "negative_embeds": ("EMBEDS", ),
            "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers"

    def sample(self, maked_pipeline, positive_embeds, negative_embeds, height, width, steps, cfg, seed):
        images = maked_pipeline[0](
            prompt_embeds=positive_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            negative_prompt_embeds=negative_embeds,
            generator=torch.Generator(self.torch_device).manual_seed(seed)
        ).images
        return (convert_images_to_tensors(images),)

# - Stream Diffusion -

class CreateIntListNode:
    @classmethod
    def INPUT_TYPES(s):
        max_element = 10
        return {
            "required": {
                "elements_count" : ("INT", {"default": 2, "min": 1, "max": max_element, "step": 1}),
            }, 
            "optional": {
                f"element_{i}": ("INT", {"default": 0}) for i in range(1, max_element)
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "create_list"

    CATEGORY = "Diffusers/StreamDiffusion"

    def create_list(self, elements_count, **kwargs):
        return ([value for key, value in kwargs.items()][:elements_count], )

class LcmLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "lora_name": (folder_paths.get_filename_list("loras"), ), }}

    RETURN_TYPES = ("LCM_LORA",)
    FUNCTION = "load_lora"

    CATEGORY = "Diffusers/StreamDiffusion"

    def load_lora(self, lora_name):
        return (load_file(folder_paths.get_full_path("loras", lora_name)), )

class StreamDiffusionCreateStream:
    def __init__(self):
        self.dtype = torch.float32
        self.torch_device = get_torch_device()
        self.tmp_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "maked_pipeline": ("MAKED_PIPELINE", ),
                "t_index_list": ("LIST", ),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "do_add_noise": ("BOOLEAN", {"default": True}),
                "use_denoising_batch": ("BOOLEAN", {"default": True}),
                "frame_buffer_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": "none"}),
                "acceleration": (["none", "xformers", "tensorrt"], {"default": "tensorrt"}),
                "lcm_lora" : ("LCM_LORA", ),
                "tiny_vae" : ("STRING", {"default": "madebyollin/taesd"}),
                "use_lcm_lora" : ("BOOLEAN", {"default": True}),
                "use_tiny_vae" : ("BOOLEAN", {"default": True}),
            }, 
        }

    RETURN_TYPES = ("STREAM",)
    FUNCTION = "load_stream"

    CATEGORY = "Diffusers/StreamDiffusion"

    def load_stream(self, maked_pipeline, t_index_list, width, height, do_add_noise, use_denoising_batch, frame_buffer_size, cfg_type, acceleration, lcm_lora, tiny_vae, use_lcm_lora, use_tiny_vae):
        model_id_or_path = maked_pipeline[1]
        maked_pipeline: StableDiffusionPipeline = copy.deepcopy(maked_pipeline[0])
        lcm_lora = copy.deepcopy(lcm_lora)
        stream = StreamDiffusion(
            pipe = maked_pipeline,
            t_index_list = t_index_list,
            torch_dtype = self.dtype,
            width = width,
            height = height,
            do_add_noise = do_add_noise,
            use_denoising_batch = use_denoising_batch,
            frame_buffer_size = frame_buffer_size,
            cfg_type = cfg_type,
        )
        stream.load_lcm_lora(lcm_lora)
        stream.fuse_lora()
        stream.vae = AutoencoderTiny.from_pretrained(
            pretrained_model_name_or_path=tiny_vae,
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        ).to(
            device=maked_pipeline.device, 
            dtype=maked_pipeline.dtype
        )
        
        batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"

                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                    )
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )

                cuda_steram = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_steram, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_steram,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")
        return (stream, )

class StreamDiffusionSampler:
    def __init__(self):
        self.torch_device = get_torch_device()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream": ("STREAM", ),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 100.0}),
                "delta": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "num": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "warmup": ("INT", {"default": 1, "min": 0, "max": 10000}),
            },
            "optional" : {
                "image" : ("IMAGE", )
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers/StreamDiffusion"

    def sample(self, stream: StreamDiffusion, positive, negative, steps, cfg, delta, seed, num, warmup, image = None):
        stream.prepare(
            prompt = positive,
            negative_prompt = negative,
            num_inference_steps = steps,
            guidance_scale = cfg,
            delta = delta,
            seed = seed
        )
        
        if image != None:
            image = convert_tensors_to_images(image)
            image = resize_images(image, (stream.width, stream.height))
            
        for _ in range(warmup):
            stream()

        result = []
        for _ in range(num):
            x_outputs = []
            if image is None:
                x_outputs.append(stream.txt2img())
            else:
                stream(image[0])
                for i in image[1:] + image[-1:]:
                    x_outputs.append(stream(i))
            for x_output in x_outputs:
                result.append(postprocess_image(x_output, output_type="pil")[0])
        
        return (convert_images_to_tensors(result),)

class StreamDiffusionWarmup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream": ("STREAM", ),
                "negative": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 100.0}),
                "delta": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "warmup": ("INT", {"default": 1, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = ("WARMUP_STREAM",)

    FUNCTION = "stream_warmup"

    CATEGORY = "Diffusers/StreamDiffusion"

    def stream_warmup(self, stream: StreamDiffusion, negative, steps, cfg, delta, seed, warmup):
        stream.prepare(
            prompt="",
            negative_prompt=negative,
            num_inference_steps = steps,
            guidance_scale = cfg,
            delta = delta,
            seed = seed
        )
        
        for _ in range(warmup):
            stream()
        
        return (stream, )


class StreamDiffusionFastSampler:  
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "warmup_stream": ("WARMUP_STREAM", ),
                "positive": ("STRING", {"multiline": True}),
                "num": ("INT", {"default": 1, "min": 1, "max": 10000}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers/StreamDiffusion"

    def sample(self, warmup_stream, positive, num):
        stream: StreamDiffusion = warmup_stream
        
        stream.update_prompt(positive)

        result = []
        for _ in range(num):
            x_output = stream.txt2img()
            result.append(postprocess_image(x_output, output_type="pil")[0])
        return (convert_images_to_tensors(result),)

# - - - - - - - - - - - - - - - - - -


NODE_CLASS_MAPPINGS = {
    "DiffusersPipelineLoader": DiffusersPipelineLoader,
    "DiffusersVaeLoader": DiffusersVaeLoader,
    "DiffusersSchedulerLoader": DiffusersSchedulerLoader,
    "DiffusersModelMakeup": DiffusersModelMakeup,
    "DiffusersClipTextEncode": DiffusersClipTextEncode,
    "DiffusersSampler": DiffusersSampler,
    "CreateIntListNode": CreateIntListNode,
    "LcmLoraLoader": LcmLoraLoader,
    "StreamDiffusionCreateStream": StreamDiffusionCreateStream,
    "StreamDiffusionSampler": StreamDiffusionSampler,
    "StreamDiffusionWarmup": StreamDiffusionWarmup,
    "StreamDiffusionFastSampler": StreamDiffusionFastSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipelineLoader": "Diffusers Pipeline Loader",
    "DiffusersVaeLoader": "Diffusers Vae Loader",
    "DiffusersSchedulerLoader": "Diffusers Scheduler Loader",
    "DiffusersModelMakeup": "Diffusers Model Makeup",
    "DiffusersClipTextEncode": "Diffusers Clip Text Encode",
    "DiffusersSampler": "Diffusers Sampler",
    "CreateIntListNode": "Create Int List",
    "LcmLoraLoader": "LCM Lora Loader",
    "StreamDiffusionCreateStream": "StreamDiffusion Create Stream",
    "StreamDiffusionSampler": "StreamDiffusion Sampler",
    "StreamDiffusionWarmup": "StreamDiffusion Warmup",
    "StreamDiffusionFastSampler": "StreamDiffusion Fast Sampler",
}
