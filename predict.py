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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = FluxPipeline.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
        transformer = FluxTransformer2DModel.from_pretrained(
            BASE_MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.pipe.transformer = transformer
        self.pipe.to(self.device)

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
        input_image: Path = Input(
            description="Input image for editing and subject conditioning"),
        inpainting_mask: Path = Input(
            description="Inpainting mask image (white areas will be inpainted)"),
        expression: str = Input(
            description="Desired facial expression for the manhwa style (e.g., 'happy', 'angry', 'surprised', 'neutral')",
            default="happy"
        ),
        height: int = Input(
            description="Height of the output image", default=768),
        width: int = Input(
            description="Width of the output image", default=512),
        seed: int = Input(description="Random seed", default=42),
        subject_lora_scale: float = Input(
            description="Scale for the Subject LoRA weights", default=1.0, ge=0.0, le=2.0),
        inpainting_lora_scale: float = Input(
            description="Scale for the Inpainting LoRA weights", default=1.0, ge=0.0, le=2.0)
    ) -> Path:

        # --- OpenAI Image Editing Step ---
        openai_edit_prompt = f"""
	Crop the face precisely and convert it into a digital illustration in {expression} style.
	Maintain exact hair style and keep eyes open.
	Pose with a slight rightward.
	Slightly widen the face while preserving original structure and strong likeness.
	Retain fine facial details—including lines and wrinkles (if present).
	For K-pop style, apply smooth skin, stylized features, and expressive eyes while maintaining resemblance.
	"""
        print(f"Constructed OpenAI Edit Prompt: {openai_edit_prompt}")

        openai_edited_image_pil = edit_image_openai(
            self.openai_client, str(input_image), openai_edit_prompt)

        if not openai_edited_image_pil:
            raise ValueError("OpenAI image editing failed. Cannot proceed.")

        # --- Load Inpainting Mask ---
        try:
            inpainting_mask_pil = PILImage.open(
                str(inpainting_mask)).convert("RGB")
            print(f"Successfully loaded inpainting mask: {inpainting_mask}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Inpainting mask file not found at {inpainting_mask}")
        except Exception as e:
            raise ValueError(
                f"Failed to load or process inpainting mask {inpainting_mask}: {e}")

        # --- Prepare LoRA Data ---
        lora_paths = [self.subject_lora_path, self.inpainting_lora_path]
        lora_weights = [[subject_lora_scale], [inpainting_lora_scale]]
        subject_images_pil = [openai_edited_image_pil]
        spatial_images_pil = [inpainting_mask_pil]

        # --- Apply LoRAs ---
        unset_lora(self.pipe.transformer)
        print(f"Applying LoRAs: {lora_paths} with weights {lora_weights}")
        set_multi_lora(self.pipe.transformer, lora_paths,
                       lora_weights=lora_weights, cond_size=768)

        # --- Flux Generation ---
        flux_prompt = "put exact face on the body, match body skin tone"
        print(f"Using fixed Flux Prompt: {flux_prompt}")

        generator = torch.Generator(self.device).manual_seed(seed)

        try:
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

            # --- Clear Cache ---
            print("Clearing attention cache.")
            clear_cache(self.pipe.transformer)

            # --- Save and Return ---
            output_path = Path(OUTPUT_FILENAME)
            generated_pil_image.save(str(output_path))
            print(f"Output image saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"Error during Flux pipeline generation: {e}")
            print("Clearing cache after error.")
            unset_lora(self.pipe.transformer)
            clear_cache(self.pipe.transformer)
            raise e
