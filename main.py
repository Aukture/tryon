import os
import argparse
from datetime import datetime
from typing import Optional, Literal, Union
from pathlib import Path
import io
import base64
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import os
from model.cloth_masker import AutoMasker, vis_mask
from model.flux.pipeline_flux_tryon import FluxTryOnPipeline
from utils import resize_and_crop, resize_and_padding
from image_utils import read_image, image_to_b64s, resize_if_necessary, improve_resolution
import os
from huggingface_hub import HfFolder
from huggingface_hub import login
from huggingface_hub import whoami
user = whoami(token=os.getenv("HF_TOKEN"))


# login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)

# Set custom cache location
# os.environ["HF_HOME"] = "/workspace/huggingface"
# os.environ["TRANSFORMERS_CACHE"] = "/workspace/model_cache"
# os.environ["DIFFUSERS_CACHE"] = "/workspace/model_cache"
# HfFolder.path = "/workspace/huggingface"

# Constants and Configuration
DEFAULT_OUTPUT_DIR = "resource/demo/output"
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_IMAGE_SIZE = (768, 1024)

# Set your custom cache paths
CUSTOM_MODEL_CACHE = "/workspace/model_cache"  # Use your large storage volume
# CUSTOM_REPO_CACHE = os.path.join(CUSTOM_MODEL_CACHE)

# Ensure directories exist
os.makedirs(CUSTOM_MODEL_CACHE, exist_ok=True)
# os.makedirs(CUSTOM_REPO_CACHE, exist_ok=True)


# Pydantic Models for Request/Response
class ImageInput(BaseModel):
    """Union type for multiple image input formats"""
    url: Optional[str] = Field(None, description="URL of the image")
    base64: Optional[str] = Field(None, description="Base64 encoded image string")
    upload: Optional[UploadFile] = Field(None, description="Uploaded image file")

    class Config:
        arbitrary_types_allowed = True

class TryOnRequest(BaseModel):
    person_image: Union[ImageInput, UploadFile] = Field(..., description="Person image in URL, base64 or file upload format")
    cloth_image: Union[ImageInput, UploadFile] = Field(..., description="Cloth image in URL, base64 or file upload format")
    # cloth_mask: Optional[Union[ImageInput, UploadFile]] = Field(None, description="Optional mask image")
    cloth_type: Literal["upper", "lower", "overall"] = Field(
        default="upper", 
        description="Type of clothing to try on"
    )
    num_inference_steps: Optional[int] = Field(
        default=50, 
        ge=10, 
        le=100, 
        description="Number of inference steps (10-100)"
    )
    guidance_scale: Optional[float] = Field(
        default=30.0, 
        ge=0.0, 
        le=50.0, 
        description="Guidance scale (0.0-50.0)"
    )
    seed: Optional[int] = Field(
        default=1250, 
        ge=-1, 
        description="Random seed (-1 for random)"
    )
    show_type: Optional[Literal["result only", "input & result", "input & mask & result"]] = Field(
        default="result only", 
        description="Type of output to show"
    )

    

class TryOnResponse(BaseModel):
    status: str = Field(..., description="Status of the operation")
    message: Optional[str] = Field(None, description="Additional message")
    result_image: Optional[str] = Field(None, description="Base64 encoded result image")
    processing_time: float = Field(..., description="Time taken in seconds")

# Initialize FastAPI app
app = FastAPI(
    title="FLUX Try-On API",
    description="API for virtual try-on using FLUX.1-Fill-dev model with support for multiple image input formats",
    version="1.0.0"
)

# Define args:
class Args:
    def __init__(self):
        self.base_model_path = "black-forest-labs/FLUX.1-Fill-dev"
        self.resume_path = "zhengchong/CatVTON"
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.mixed_precision = "bf16"
        self.allow_tf32 = True
        self.width = 768
        self.height = 1024

# Command line arguments parser
# def parse_args():
#     parser = argparse.ArgumentParser(description="FLUX Try-On API")
#     parser.add_argument(
#         "--base_model_path",
#         type=str,
#         default="black-forest-labs/FLUX.1-Fill-dev",
#         help="The path to the base model to use for evaluation."
#     )
#     parser.add_argument(
#         "--resume_path",
#         type=str,
#         default="zhengchong/CatVTON",
#         help="The Path to the checkpoint of trained tryon model."
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default=DEFAULT_OUTPUT_DIR,
#         help="The output directory where the model predictions will be written."
#     )
#     parser.add_argument(
#         "--mixed_precision",
#         type=str,
#         default="bf16",
#         choices=["no", "fp16", "bf16"],
#         help="Whether to use mixed precision."
#     )
#     parser.add_argument(
#         "--allow_tf32",
#         action="store_true",
#         default=True,
#         help="Whether or not to allow TF32 on Ampere GPUs."
#     )
#     parser.add_argument(
#         "--width",
#         type=int,
#         default=768,
#         help="The width of the input image."
#     )
#     parser.add_argument(
#         "--height",
#         type=int,
#         default=1024,
#         help="The height of the input image."
#     )
#     return parser.parse_args()

# Initialize global variables
args = Args() #parse_args()
pipeline_flux = None
mask_processor = None
automasker = None

# Helper functions
async def process_image_input(image_input: Union[ImageInput, UploadFile], is_mask: bool = False) -> Image.Image:
    """Process image input from various formats (URL, base64, or file upload)"""
    try:
        pil_image = Image.Image()
        if isinstance(image_input, UploadFile):
            # Handle direct file upload
            validate_image_file(image_input)
            image_bytes = await image_input.read()
            pil_image = read_image(image_bytes, isCV2=False)
        else:
            # Handle ImageInput object
            if image_input.url:
                pil_image = read_image(image_input.url, isCV2=False)
            elif image_input.base64:
                pil_image = read_image(image_input.base64, isCV2=False)
            elif image_input.upload:
                validate_image_file(image_input.upload)
                image_bytes = await image_input.upload.read()
                pil_image = read_image(image_bytes, isCV2=False)
            else:
                raise ValueError("No valid image input provided")

        # Convert to RGB if needed
        if pil_image and pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Resize if necessary
        if not is_mask:
            pil_image = resize_if_necessary(pil_image)
            pil_image = improve_resolution(pil_image, return_pil=True, size=DEFAULT_IMAGE_SIZE)

        return pil_image

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image input: {str(e)}"
        )

def validate_image_file(file: UploadFile):
    """Validate uploaded image file"""
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported types are {SUPPORTED_IMAGE_TYPES}"
        )
    
    if  file.size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_IMAGE_SIZE/1024/1024}MB"
        )

def image_grid(imgs, rows, cols):
    """Create a grid of images."""
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# APP startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and resources when the app starts."""
    global pipeline_flux, mask_processor, automasker
    
    try:
        # Load models
        # repo_path = snapshot_download(repo_id=args.resume_path)
        # Download with custom cache
        repo_path = snapshot_download(
            repo_id=args.resume_path,
            cache_dir=CUSTOM_MODEL_CACHE,
            local_dir=CUSTOM_MODEL_CACHE,
            local_dir_use_symlinks=False  # Avoid symlinks that might cause space issues
        )
        # pipeline_flux = FluxTryOnPipeline.from_pretrained(args.base_model_path)
        # Load pipeline with custom paths
        pipeline_flux = FluxTryOnPipeline.from_pretrained(
            args.base_model_path,
            cache_dir=CUSTOM_MODEL_CACHE,
            subfolder=None,
            local_files_only=False
        )
        pipeline_flux.load_lora_weights(
            os.path.join(repo_path, "flux-lora"), 
            weight_name='pytorch_lora_weights.safetensors'
        )
        pipeline_flux.to("cuda", torch.bfloat16)

        # Initialize AutoMasker
        mask_processor = VaeImageProcessor(
            vae_scale_factor=8, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True
        )
        automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device='cuda'
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize models: {str(e)}")

@app.post("/try-on", response_model=TryOnResponse)
async def virtual_try_on(
    person_image: Union[ImageInput, UploadFile] = File(...),
    cloth_image: Union[ImageInput, UploadFile] = File(...),
    cloth_mask: Optional[Union[ImageInput, UploadFile]] = File(None),
    cloth_type: str = Form("upper"),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(30.0),
    seed: int = Form(42),
    show_type: str = Form("input & mask & result")
):
    """Endpoint for virtual try-on with support for multiple image input formats."""
    start_time = datetime.now()
    global pipeline_flux, mask_processor, automasker

    try:
        # Process input images
        person_img = await process_image_input(person_image)
        cloth_img = await process_image_input(cloth_image)
        
        # Process mask if provided
        mask = None
        if cloth_mask:
            mask = await process_image_input(cloth_mask, is_mask=True)
            if len(np.unique(np.array(mask))) == 1:
                mask = None
            else:
                mask = np.array(mask)
                mask[mask > 0] = 255
                mask = Image.fromarray(mask)

        # Set random seed
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda').manual_seed(seed)

        # Adjust image sizes
        person_img = resize_and_crop(person_img, (args.width, args.height))
        cloth_img = resize_and_padding(cloth_img, (args.width, args.height))

        # Process mask
        if mask is not None:
            mask = resize_and_crop(mask, (args.width, args.height))
        else:
            mask = automasker(person_img, cloth_type)['mask']
        mask = mask_processor.blur(mask, blur_factor=9)

        # Inference
        result_img = pipeline_flux(
            image=person_img,
            condition_image=cloth_img,
            mask_image=mask,
            height=args.height,
            width=args.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Post-processing
        masked_person = vis_mask(person_img, mask)

        # Generate output based on show type``
        if show_type == "result only":
            final_result = result_img
        else:
            width, height = person_img.size
            if show_type == "input & result":
                condition_width = width // 2
                conditions = image_grid([person_img, cloth_img], 2, 1)
            else:
                condition_width = width // 3
                conditions = image_grid([person_img, masked_person, cloth_img], 3, 1)
            
            conditions = conditions.resize((condition_width, height), Image.NEAREST)
            final_result = Image.new("RGB", (width + condition_width + 5, height))
            final_result.paste(conditions, (0, 0))
            final_result.paste(result_img, (condition_width + 5, 0))

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Convert to base64
        result_base64 = image_to_b64s(final_result)

        return TryOnResponse(
            status="success",
            message="Try-on completed successfully",
            result_image=result_base64,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during try-on processing: {str(e)}"
        )

@app.post("/try-on-file", response_class=StreamingResponse)
async def virtual_try_on_file(
    person_image: Union[ImageInput, UploadFile] = File(...),
    cloth_image: Union[ImageInput, UploadFile] = File(...),
    cloth_mask: Optional[Union[ImageInput, UploadFile]] = File(None),
    cloth_type: str = Form("upper"),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(30.0),
    seed: int = Form(42),
    show_type: str = Form("input & mask & result")
):
    """Endpoint that returns the try-on result as a file attachment"""
    try:
        # Call the main try-on function
        result = await virtual_try_on(
            person_image=person_image,
            cloth_image=cloth_image,
            cloth_mask=cloth_mask,
            cloth_type=cloth_type,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            show_type=show_type
        )

        # Convert base64 back to bytes
        image_bytes = base64.b64decode(result.result_image)
        
        # Create a streaming response
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "attachment; filename=tryon_result.jpg",
                "X-Processing-Time": str(result.processing_time)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating file response: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={"status": "healthy"},
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
