import copy
import os
import torch
from safetensors.torch import load_file
from .utils import SCHEDULERS, token_auto_concat_embeds, vae_pt_to_vae_diffuser, convert_images_to_tensors, convert_tensors_to_images, resize_images
from comfy.model_management import get_torch_device
import folder_paths
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoencoderTiny


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
        pipeline = pipeline[0]
        pipeline.vae = autoencoder
        pipeline.scheduler = scheduler
        pipeline.safety_checker = None if pipeline.safety_checker is None else lambda images, **kwargs: (images, [False])
        pipeline.enable_attention_slicing()
        pipeline = pipeline.to(self.torch_device)
        return (pipeline,)

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
        positive_embeds, negative_embeds = token_auto_concat_embeds(maked_pipeline, positive,negative)

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
        images = maked_pipeline(
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
                "xformers_memory_efficient_attention": ("BOOLEAN", {"default": False}),
                "lcm_lora" : ("LCM_LORA", ),
                "tiny_vae" : ("STRING", {"default": "madebyollin/taesd"})
            }, 
        }

    RETURN_TYPES = ("STREAM",)
    FUNCTION = "load_stream"

    CATEGORY = "Diffusers/StreamDiffusion"

    def load_stream(self, maked_pipeline, t_index_list, width, height, do_add_noise, use_denoising_batch, frame_buffer_size, cfg_type, xformers_memory_efficient_attention, lcm_lora, tiny_vae):
        maked_pipeline = copy.deepcopy(maked_pipeline)
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
        
        if xformers_memory_efficient_attention:
            maked_pipeline.enable_xformers_memory_efficient_attention()
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
