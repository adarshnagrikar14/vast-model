import os
import torch
import io
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image as PILImage
from huggingface_hub import login
from src.pipeline import FluxPipeline
from src.lora_helper import set_single_lora, set_multi_lora, unset_lora
from src.transformer_flux import FluxTransformer2DModel
import tempfile
import traceback

import sys
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

load_dotenv()
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

# Read OpenAI API Key
openai_api_key = os.environ.get("OPENAI_API_KEY")

BASE_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
LORA_BASE_PATH = "./models"
SUBJECT_LORA_FILENAME = "subject.safetensors"
INPAINTING_LORA_FILENAME = "inpainting.safetensors"
TEMP_DIR = "./tmp_job_files"


def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        if hasattr(attn_processor, 'bank_kv'):
            attn_processor.bank_kv.clear()


class Predictor:
    def setup(self):
        """Initializes the prediction pipeline and checks for LoRA files."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Predictor using device: {self.device}")

        # Load models
        try:
            self.pipe = FluxPipeline.from_pretrained(
                BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
            transformer = FluxTransformer2DModel.from_pretrained(
                BASE_MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)
            self.pipe.transformer = transformer
            self.pipe.to(self.device)
            print("Flux pipeline and transformer loaded successfully.")
        except Exception as e:
            print(f"FATAL: Failed to load models from {BASE_MODEL_PATH}: {e}")
            raise  # Stop initialization if models can't load

        # Configure OpenAI Client
        self.openai_client = None
        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                print("OpenAI client configured successfully.")
            except Exception as e:
                print(f"Warning: Failed to configure OpenAI client: {e}")
        else:
            print(
                "Warning: OPENAI_API_KEY environment variable not set. OpenAI editing will be disabled.")

        # Check if LoRA files exist during setup
        self.subject_lora_path = os.path.join(
            LORA_BASE_PATH, SUBJECT_LORA_FILENAME)
        self.inpainting_lora_path = os.path.join(
            LORA_BASE_PATH, INPAINTING_LORA_FILENAME)
        if not os.path.exists(self.subject_lora_path):
            raise FileNotFoundError(
                f"Subject LoRA file not found at {self.subject_lora_path}")
        if not os.path.exists(self.inpainting_lora_path):
            raise FileNotFoundError(
                f"Inpainting LoRA file not found at {self.inpainting_lora_path}")
        print("Subject and Inpainting LoRA files found.")

    def predict(
        self,
        input_image_bytes: bytes,   # Expect bytes
        mask_image_bytes: bytes,    # Expect bytes
        expression: str = "happy",
        height: int = 768,
        width: int = 512,
        seed: int = 42,
        subject_lora_scale: float = 1.0,
        inpainting_lora_scale: float = 1.0
    ) -> str:  # Still returns path to the generated file
        """
        Generates an image using the Flux pipeline with specified LoRAs.
        Accepts input images as bytes.
        Returns the path to a temporary output PNG file.
        """
        print(
            f"Predictor received job with bytes: expr={expression}, seed={seed}, size={width}x{height}, subj_lora={subject_lora_scale}, inp_lora={inpainting_lora_scale}")
        output_path_str = None

        try:
            # --- Load Input Images from Bytes ---
            try:
                # Use io.BytesIO to allow PIL to read from memory buffer
                subject_image_pil = PILImage.open(
                    io.BytesIO(input_image_bytes)).convert("RGB")
                print(
                    f"Loaded subject image from bytes ({len(input_image_bytes)} bytes).")
            except Exception as e:
                # More specific error for byte loading failure
                raise ValueError(
                    f"Failed to load subject image from bytes: {e}")

            try:
                inpainting_mask_pil = PILImage.open(
                    io.BytesIO(mask_image_bytes)).convert("RGB")
                print(
                    f"Loaded inpainting mask from bytes ({len(mask_image_bytes)} bytes).")
            except Exception as e:
                raise ValueError(
                    f"Failed to load inpainting mask from bytes: {e}")

            # --- Prepare LoRA Data ---
            lora_paths = [self.subject_lora_path, self.inpainting_lora_path]
            lora_weights = [[subject_lora_scale], [inpainting_lora_scale]]
            subject_images_pil = [subject_image_pil]
            spatial_images_pil = [inpainting_mask_pil]

            # --- Apply LoRAs and Generate ---
            unset_lora(self.pipe.transformer)
            print(f"Applying LoRAs...")
            set_multi_lora(self.pipe.transformer, lora_paths,
                           lora_weights=lora_weights, cond_size=768)

            flux_prompt = "put exact face on the body, match body skin tone"
            print(f"Using Flux Prompt: '{flux_prompt}'")
            generator = torch.Generator(self.device).manual_seed(seed)

            print("Starting Flux pipeline generation...")
            generated_pil_image = self.pipe(
                prompt=flux_prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=25,
                max_sequence_length=512,
                generator=generator,
                subject_images=subject_images_pil,
                spatial_images=spatial_images_pil,
                cond_size=768,
            ).images[0]
            print("Flux pipeline generation successful.")

            # --- Save Output Temporarily ---
            # Output saving still uses a temp file path
            output_fd, output_path_str = tempfile.mkstemp(
                suffix=".png", prefix="flux_output_", dir=TEMP_DIR)
            os.close(output_fd)
            generated_pil_image.save(output_path_str)
            print(f"Predictor output saved temporarily to {output_path_str}")
            return output_path_str

        except Exception as e:
            print(f"ERROR during prediction: {e}\n{traceback.format_exc()}")
            if output_path_str:
                print(f"(Attempted output path was: {output_path_str})")
            raise

        finally:
            # --- Cleanup GPU state ---
            print("Unsetting LoRA and clearing cache.")
            unset_lora(self.pipe.transformer)
            clear_cache(self.pipe.transformer)
            # Output file cleanup is handled by QueueManager
