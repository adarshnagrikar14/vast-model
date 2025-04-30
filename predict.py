import os
import torch
import requests
import io
import base64
from dotenv import load_dotenv
from PIL import Image as PILImage
from huggingface_hub import login
from openai import OpenAI
from src.pipeline import FluxPipeline
from cog import BasePredictor, Input, Path
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
OUTPUT_FILENAME = "output.png"


def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        if hasattr(attn_processor, 'bank_kv'):
            attn_processor.bank_kv.clear()


def edit_image_openai(client: OpenAI, input_image_path: str, edit_prompt: str) -> PILImage.Image | None:
    """
    Uses OpenAI's image edit API to modify an image, handling b64_json response.

    Args:
        client: Initialized OpenAI client.
        input_image_path: Path to the input image file.
        edit_prompt: The prompt describing the edit.

    Returns:
        A PIL Image object of the edited image, or None on error.
    """
    if not client:
        print("Error: OpenAI client not configured. Set OPENAI_API_KEY.")
        return None
    try:
        print(
            f"Sending image '{input_image_path}' to OpenAI for editing with prompt: '{edit_prompt}'")
        with open(input_image_path, "rb") as img_file:
            response = client.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=edit_prompt,
                n=1,
                size="1024x1024",
                quality="low",
            )

        # Extract Base64 encoded image data
        b64_data = response.data[0].b64_json
        if not b64_data:
            print("Error: OpenAI response did not contain b64_json data.")
            return None

        print("Received b64_json data from OpenAI. Decoding...")

        # Decode the Base64 string to bytes
        image_bytes = base64.b64decode(b64_data)

        # Create PIL image from bytes
        edited_image_pil = PILImage.open(
            io.BytesIO(image_bytes)).convert("RGB")
        print("Successfully created PIL image from OpenAI b64_json response.")
        return edited_image_pil

    except FileNotFoundError:
        print(f"Error: Input image file not found at {input_image_path}")
        return None
    except AttributeError:
        print("Error: Could not find 'b64_json' in OpenAI response data.")
        return None
    except base64.binascii.Error as b64_error:
        print(f"Error decoding Base64 data from OpenAI: {b64_error}")
        return None
    except Exception as e:
        # Catch potential API errors from OpenAI library itself
        print(
            f"An error occurred during OpenAI image editing or processing: {e}")
        return None


class Predictor(BasePredictor):
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
        input_image: str,           # Expect string path from queue manager
        inpainting_mask: str,       # Expect string path from queue manager
        expression: str = "happy",  # Default values should match queue manager's defaults
        height: int = 768,
        width: int = 512,
        seed: int = 42,
        subject_lora_scale: float = 1.0,
        inpainting_lora_scale: float = 1.0
    ) -> str:  # Return the path (string) to the generated file
        """
        Generates an image using the Flux pipeline with specified LoRAs.
        Assumes input_image is pre-processed (e.g., OpenAI edited) if necessary.
        Returns the path to a temporary output PNG file.
        """
        print(
            f"Predictor received job: expr={expression}, seed={seed}, size={width}x{height}, subj_lora={subject_lora_scale}, inp_lora={inpainting_lora_scale}")
        output_path_str = None  # Define here for potential use in logging errors

        try:
            # --- Load Input Images ---
            # Input paths are now strings passed by the queue manager
            try:
                subject_image_pil = PILImage.open(input_image).convert("RGB")
                # print(f"Loaded subject image: {input_image}") # Optional logging
            except Exception as e:
                raise ValueError(
                    f"Failed to load subject image {input_image}: {e}")

            try:
                inpainting_mask_pil = PILImage.open(
                    inpainting_mask).convert("RGB")
                # print(f"Loaded inpainting mask: {inpainting_mask}") # Optional logging
            except Exception as e:
                raise ValueError(
                    f"Failed to load inpainting mask {inpainting_mask}: {e}")

            # --- Prepare LoRA Data ---
            lora_paths = [self.subject_lora_path, self.inpainting_lora_path]
            lora_weights = [[subject_lora_scale], [inpainting_lora_scale]]
            subject_images_pil = [subject_image_pil]
            spatial_images_pil = [inpainting_mask_pil]

            # --- Apply LoRAs and Generate ---
            # Ensure clean state before applying
            unset_lora(self.pipe.transformer)
            print(f"Applying LoRAs...")  # Simplified logging
            # Assuming fixed cond_size
            set_multi_lora(self.pipe.transformer, lora_paths,
                           lora_weights=lora_weights, cond_size=768)

            flux_prompt = "put exact face on the body, match body skin tone"  # Fixed prompt
            print(f"Using Flux Prompt: '{flux_prompt}'")
            generator = torch.Generator(self.device).manual_seed(seed)

            print("Starting Flux pipeline generation...")
            # Ensure pipeline arguments match expected names and types
            generated_pil_image = self.pipe(
                prompt=flux_prompt,
                height=height,
                width=width,
                guidance_scale=3.5,         # Keep fixed or make parameter if needed
                num_inference_steps=25,     # Keep fixed or make parameter if needed
                max_sequence_length=512,    # Keep fixed or make parameter if needed
                generator=generator,
                subject_images=subject_images_pil,
                spatial_images=spatial_images_pil,
                cond_size=768,              # Keep fixed or make parameter if needed
            ).images[0]
            print("Flux pipeline generation successful.")

            # --- Save Output Temporarily ---
            # Use mkstemp to get a unique temporary file name
            output_fd, output_path_str = tempfile.mkstemp(
                suffix=".png", prefix="flux_output_")
            os.close(output_fd)  # Close the file descriptor
            generated_pil_image.save(output_path_str)
            print(f"Predictor output saved temporarily to {output_path_str}")
            return output_path_str  # Return the path as a string

        except Exception as e:
            print(f"ERROR during prediction: {e}\n{traceback.format_exc()}")
            # If output_path was created before error, log it for potential manual cleanup debug
            if output_path_str:
                print(f"(Attempted output path was: {output_path_str})")
            # Queue manager will handle cleanup of input files based on the exception
            raise  # Re-raise the exception for the queue manager

        finally:
            # --- Cleanup GPU state ---
            # Always unset LoRA and clear cache, regardless of success/failure
            print("Unsetting LoRA and clearing cache.")
            unset_lora(self.pipe.transformer)
            clear_cache(self.pipe.transformer)
            # NOTE: Cleanup of the output file (output_path_str) is now handled
            # by the QueueManager after it reads the file or encounters an error.
