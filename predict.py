import os
import io
import sys
import base64
import tempfile
import traceback
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image as PILImage
from huggingface_hub import login
import google.generativeai as genai
import torch
import json

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.pipeline import FluxPipeline
from src.lora_helper import set_multi_lora, unset_lora
from src.transformer_flux import FluxTransformer2DModel

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

load_dotenv()
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_key2 = os.environ.get("OPENAI_API_KEY2")
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Configure Google Gemini API if key is available
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    print("Warning: GOOGLE_API_KEY environment variable not set. Gemini processing will be disabled.")

# OpenAI API keys rotation
OPENAI_API_KEYS = []
if openai_api_key:
    OPENAI_API_KEYS.append(openai_api_key)
if openai_api_key2:
    OPENAI_API_KEYS.append(openai_api_key2)

# Current OpenAI key index
CURRENT_OPENAI_KEY_INDEX = 0

# Function to get the next OpenAI API key


def get_next_openai_key():
    global CURRENT_OPENAI_KEY_INDEX
    if not OPENAI_API_KEYS:
        return None
    key = OPENAI_API_KEYS[CURRENT_OPENAI_KEY_INDEX]
    CURRENT_OPENAI_KEY_INDEX = (
        CURRENT_OPENAI_KEY_INDEX + 1) % len(OPENAI_API_KEYS)
    return key


LORA_BASE_PATH = "./models"
TEMP_DIR = "./tmp_job_files"
SUBJECT_LORA_FILENAME = "subject.safetensors"
BASE_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
INPAINTING_LORA_FILENAME = "inpainting.safetensors"

DEFAULT_SEED = 42
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 768
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_NUM_INFERENCE_STEPS = 25
DEFAULT_SUBJECT_LORA_SCALE = 0.85
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_INPAINTING_LORA_SCALE = 0.85

try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except OSError as e:
    print(f"FATAL: Could not create temporary directory {TEMP_DIR}: {e}")

# Gemini API keys rotation
GEMINI_API_KEYS = [
    "AIzaSyDHBQ5_6C6emHD7aUO_eWTzCE7wvIxrNvA",
    "AIzaSyAxqRhLegXHms1nCmzhuaQQzp-a_HPXUfY",
    "AIzaSyCRWFSaTIOAGYPMlVa2VbPoaz4knoFkj-E",
    "AIzaSyBIG8QfWK5v77pVE3K3orGwq1Y5oRQkjqs",
    "AIzaSyBM3TyRgeaHhs7cqlRR5aJHF97tNu-9jX4",
    "AIzaSyDRpN6hf44xwjMwhFoclqUNBJ2IZyZzeuQ",
    "AIzaSyDmVm1yhhPRnprwM1X1SRrj7ntVNjLu8X8",
    "AIzaSyDlsFkI-SFClohprrj930BTBWVMwZ4VCR0",
    "AIzaSyBw8j9sjcwuK-pHh0qusfkmla5IGrnoeiE",
    "AIzaSyD_X94Gt0cENkE6BJTU54HcmrWO_6vgKUA",
    "AIzaSyCWEaGGNQKW_8Rv9Nl2k9sbOQgkTWvOOVo",
    "AIzaSyDBHUSRpvLTE0LBWkufuB8jBWo1sXP9-ms",
    "AIzaSyDBHUSRpvLTE0LBWkufuB8jBWo1sXP9-ms"
]
# Current key index
CURRENT_KEY_INDEX = 0

# Function to get the next API key


def get_next_gemini_key():
    global CURRENT_KEY_INDEX
    key = GEMINI_API_KEYS[CURRENT_KEY_INDEX]
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(GEMINI_API_KEYS)
    return key


def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        if hasattr(attn_processor, 'bank_kv'):
            attn_processor.bank_kv.clear()


def edit_image_openai(client: OpenAI, input_image_path: str, edit_prompt: str) -> PILImage.Image | None:
    if not client:
        return None
    try:
        with open(input_image_path, "rb") as img_file:
            response = client.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=edit_prompt,
                n=1,
                size="1024x1024",
                quality="low"
            )

        b64_data = response.data[0].b64_json
        if not b64_data:
            return None

        image_bytes = base64.b64decode(b64_data)
        edited_image_pil = PILImage.open(
            io.BytesIO(image_bytes)).convert("RGB")
        return edited_image_pil

    except FileNotFoundError:
        return None
    except AttributeError:
        return None
    except base64.binascii.Error as b64_error:
        return None
    except Exception as e:
        print(
            f"An error occurred during OpenAI image editing: {e}\n{traceback.format_exc()}")
        return None


def image_bytes_to_data_uri(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Converts image bytes to a base64 data URI."""
    base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def process_replicate(input_image_bytes, mask_image_bytes, expression="k-pop happy"):
    """Process a job using replicate"""
    # Check if expression is a number and convert to string from prompt.json
    if expression in ["0", "1", "2", "3"]:
        try:
            with open("prompt.json", "r") as f:
                prompts = json.load(f)
            expression = prompts.get(expression, "k-pop happy")
        except Exception as e:
            print(f"Error loading expression from prompt.json: {e}")

    try:
        import requests
        import tempfile
        import replicate

        input_image_uri = image_bytes_to_data_uri(input_image_bytes)
        mask_image_uri = image_bytes_to_data_uri(mask_image_bytes)

        output_url = replicate.run(
            "adarshnagrikar14/manhwa-ai:2cfc253b2070af68f932afced7959b05de1e4ec800200ef945f338d3c289a8a1",
            input={
                "seed": DEFAULT_SEED,
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "expression": expression,
                "input_image": input_image_uri,
                "inpainting_mask": mask_image_uri,
                "subject_lora_scale": DEFAULT_SUBJECT_LORA_SCALE,
                "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
                "inpainting_lora_scale": DEFAULT_INPAINTING_LORA_SCALE,
            }
        )

        # Download image from URL
        response = requests.get(output_url, stream=True)
        response.raise_for_status()

        # Save to temp file
        output_fd, output_path_str = tempfile.mkstemp(
            suffix=".png", prefix="replicate_output_", dir=TEMP_DIR)
        with os.fdopen(output_fd, 'wb') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)

        # Read and encode the image
        with open(output_path_str, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Clean up
        try:
            os.unlink(output_path_str)
        except Exception as e:
            print(
                f"Warning: Failed to delete temp output file {output_path_str}: {e}")

        return image_base64

    except Exception as e:
        print(f"Error in Replicate processing: {e}\n{traceback.format_exc()}")
        raise


class Predictor:
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.pipe = FluxPipeline.from_pretrained(
                BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
            transformer = FluxTransformer2DModel.from_pretrained(
                BASE_MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)
            self.pipe.transformer = transformer
            self.pipe.to(self.device)
        except Exception as e:
            print(f"FATAL: Failed to load models from {BASE_MODEL_PATH}: {e}")
            raise

        self.openai_client = None
        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
            except Exception as e:
                print(f"Warning: Failed to configure OpenAI client: {e}")
        else:
            print(
                "Warning: OPENAI_API_KEY environment variable not set. OpenAI editing will be disabled.")

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

    def predict(
        self,
        input_image_bytes: bytes,
        mask_image_bytes: bytes,
        expression: str = "k-pop happy",
    ) -> str:
        output_path_str = None
        original_temp_path = None
        png_temp_path = None

        try:
            try:
                original_temp_fd, original_temp_path = tempfile.mkstemp(
                    suffix=".tmp", dir=TEMP_DIR)
                with os.fdopen(original_temp_fd, 'wb') as tmp_file:
                    tmp_file.write(input_image_bytes)

                needs_conversion = False
                try:
                    with PILImage.open(original_temp_path) as img:
                        if img.format != 'PNG':
                            needs_conversion = True
                except Exception as img_err:
                    needs_conversion = True

                openai_input_path = original_temp_path
                if needs_conversion and self.openai_client:
                    try:
                        img = PILImage.open(original_temp_path)
                        png_fd, png_temp_path = tempfile.mkstemp(
                            suffix=".png", dir=TEMP_DIR)
                        os.close(png_fd)
                        img.save(png_temp_path, format='PNG')
                        openai_input_path = png_temp_path
                    except Exception as conv_e:
                        print(
                            f"Warning: Failed to convert subject to PNG: {conv_e}.")

            except Exception as e:
                if original_temp_path and os.path.exists(original_temp_path):
                    os.unlink(original_temp_path)
                if png_temp_path and os.path.exists(png_temp_path):
                    os.unlink(png_temp_path)
                raise ValueError(f"Failed processing input image bytes: {e}")

            fixed_expression = expression
            openai_edit_prompt = f"""
            - Crop the face precisely (looking stratight) and convert it into a digital illustration in {fixed_expression} style.
            - It should be a digital illustration.
            - Maintain facial details, resemblance, and hair style and keep eyes open.
            - Retain fine facial details—including lines and wrinkles (if present).
            """

            edited_subject_pil = edit_image_openai(
                self.openai_client, openai_input_path, openai_edit_prompt)

            if edited_subject_pil:
                subject_image_pil = edited_subject_pil
            else:
                try:
                    subject_image_pil = PILImage.open(
                        original_temp_path).convert("RGB")
                except Exception as e:
                    if original_temp_path and os.path.exists(original_temp_path):
                        os.unlink(original_temp_path)
                    if png_temp_path and os.path.exists(png_temp_path):
                        os.unlink(png_temp_path)
                    raise ValueError(
                        f"Failed to load original subject image from {original_temp_path}: {e}")

            if original_temp_path and os.path.exists(original_temp_path):
                try:
                    os.unlink(original_temp_path)
                except Exception as e:
                    print(
                        f"Warning: Failed to delete original temp file {original_temp_path}: {e}")
            if png_temp_path and os.path.exists(png_temp_path):
                try:
                    os.unlink(png_temp_path)
                except Exception as e:
                    print(
                        f"Warning: Failed to delete converted PNG temp file {png_temp_path}: {e}")

            try:
                inpainting_mask_pil = PILImage.open(
                    io.BytesIO(mask_image_bytes)).convert("RGB")
            except Exception as e:
                raise ValueError(
                    f"Failed to load inpainting mask from bytes: {e}")

            lora_paths = [self.subject_lora_path, self.inpainting_lora_path]
            lora_weights = [[DEFAULT_SUBJECT_LORA_SCALE],
                            [DEFAULT_INPAINTING_LORA_SCALE]]
            subject_images_pil = [subject_image_pil]
            spatial_images_pil = [inpainting_mask_pil]

            unset_lora(self.pipe.transformer)
            set_multi_lora(self.pipe.transformer, lora_paths,
                           lora_weights=lora_weights, cond_size=768)

            flux_prompt = "put exact face on the body, match body skin tone. face should be looking straight"
            generator = torch.Generator(self.device).manual_seed(DEFAULT_SEED)

            generated_pil_image = self.pipe(
                prompt=flux_prompt,
                height=DEFAULT_HEIGHT,
                width=DEFAULT_WIDTH,
                guidance_scale=DEFAULT_GUIDANCE_SCALE,
                num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
                generator=generator,
                subject_images=subject_images_pil,
                spatial_images=spatial_images_pil,
                cond_size=768,
            ).images[0]

            output_fd, output_path_str = tempfile.mkstemp(
                suffix=".png", prefix="flux_output_", dir=TEMP_DIR)
            os.close(output_fd)
            generated_pil_image.save(output_path_str)
            return output_path_str

        except Exception as e:
            print(f"ERROR during prediction: {e}\n{traceback.format_exc()}")
            if original_temp_path and os.path.exists(original_temp_path):
                try:
                    os.unlink(original_temp_path)
                except Exception:
                    pass
            if png_temp_path and os.path.exists(png_temp_path):
                try:
                    os.unlink(png_temp_path)
                except Exception:
                    pass
            if output_path_str:
                print(f"(Attempted output path was: {output_path_str})")
            raise

        finally:
            # unset_lora(self.pipe.transformer)
            clear_cache(self.pipe.transformer)
            if original_temp_path and os.path.exists(original_temp_path):
                try:
                    os.unlink(original_temp_path)
                except Exception:
                    pass
            if png_temp_path and os.path.exists(png_temp_path):
                try:
                    os.unlink(png_temp_path)
                except Exception:
                    pass


def process_local(input_image_bytes, mask_image_bytes, expression="k-pop happy"):
    """Process a job using local resources"""
    # Check if expression is a number and convert to string from prompt.json
    if expression in ["0", "1", "2", "3"]:
        try:
            with open("prompt.json", "r") as f:
                prompts = json.load(f)
            expression = prompts.get(expression, "k-pop happy")
        except Exception as e:
            print(f"Error loading expression from prompt.json: {e}")

    predictor = Predictor()
    predictor.setup()

    output_image_path = predictor.predict(
        input_image_bytes=input_image_bytes,
        mask_image_bytes=mask_image_bytes,
        expression=expression
    )

    if output_image_path and os.path.exists(output_image_path):
        with open(output_image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        try:
            os.unlink(output_image_path)
        except Exception as e:
            print(
                f"Warning: Failed to delete temp output file {output_image_path}: {e}")
        return image_base64
    else:
        raise Exception("Prediction finished but output file not found.")


def process_gemini(input_image_bytes, mask_image_bytes, expression="k-pop happy"):
    """Process a job using Google's Gemini API"""

    # Determine the style expression for prompts and the key for desc.json
    style_for_prompts = expression  # Default for non-numeric or if prompt.json fails/key not found
    key_for_desc = expression       # Default key for desc.json

    if expression in ["0", "1", "2", "3"]:
        # Input is a numeric code. This code is the primary key for desc.json.
        key_for_desc = expression
        try:
            with open("prompt.json", "r") as f:
                prompts = json.load(f)
            # Get the textual style from prompt.json for use in Gemini/OpenAI prompts
            # Fallback to "k-pop happy" if key not in prompt.json or if prompt.json loading failed earlier
            style_for_prompts = prompts.get(expression, "k-pop happy")
        except Exception as e:
            print(f"Error loading expression from prompt.json: {e}")
            # If prompt.json loading fails, use a default textual style for prompts
            style_for_prompts = "k-pop happy"
    # else: expression is already a textual style like "k-pop happy".
    # style_for_prompts remains expression.
    # key_for_desc remains expression (will likely lead to default_description from desc.json, which is fine).

    try:
        # No longer need to check for a single API key
        # Save mask image to temp file
        mask_fd, mask_path = tempfile.mkstemp(
            suffix=".png", prefix="gemini_mask_", dir=TEMP_DIR)
        with os.fdopen(mask_fd, 'wb') as tmp_file:
            tmp_file.write(mask_image_bytes)

        # Save input image to temp file
        input_fd, input_path = tempfile.mkstemp(
            suffix=".png", prefix="gemini_input_", dir=TEMP_DIR)
        with os.fdopen(input_fd, 'wb') as tmp_file:
            tmp_file.write(input_image_bytes)

        # Load images as PIL objects
        mask_image_pil = PILImage.open(mask_path).convert("RGB")
        input_image_pil = PILImage.open(input_path).convert("RGB")

        system_prompt = """
        You are a helpful assistant skilled at generating image descriptions. You will receive a request to describe an image and you should provide a single paragraph description suitable for use with a text-to-image AI model. Be detailed and descriptive, capturing the overall style, key elements, and atmosphere of the image. Do not mention the absence of facial features or any masking of the face. Focus on clothing, pose, background, and artistic style.
        """

        # Convert mask image to bytes for Gemini
        mask_byte_arr = io.BytesIO()
        mask_image_pil.save(mask_byte_arr, format='PNG')
        mask_byte_arr = mask_byte_arr.getvalue()

        # Create image part for Gemini with the mask image
        image_part = {"mime_type": "image/png", "data": mask_byte_arr}

        # Generate image description using Gemini with key rotation
        # Try up to 5 keys before giving up
        keys_tried = 0
        max_keys_to_try = 12
        image_description = None

        while keys_tried < max_keys_to_try and image_description is None:
            try:
                # Get the next API key and configure Gemini
                api_key = get_next_gemini_key()
                print(
                    f"Trying Gemini API key {keys_tried+1}/{max_keys_to_try}")
                genai.configure(api_key=api_key)

                model = genai.GenerativeModel('gemini-2.0-flash')

                image_description_response = model.generate_content([
                    system_prompt,
                    f"{system_prompt} Describe the following image in detail:",
                    image_part
                ])
                image_description = image_description_response.text
                print(
                    f"Successfully generated image description using API key attempt {keys_tried+1}")

            except Exception as e:
                keys_tried += 1
                print(f"Error with Gemini API key attempt {keys_tried}: {e}")
                if keys_tried >= max_keys_to_try:
                    print(
                        f"Exhausted {max_keys_to_try} Gemini API key attempts")

        # If all API keys failed, fall back to desc.json
        if image_description is None:
            default_description = "A character in a dynamic pose with stylized features."
            try:
                with open("desc.json", "r") as f:
                    descriptions = json.load(f)
                # Use key_for_desc to look up in desc.json
                image_description = descriptions.get(
                    key_for_desc, default_description)
                print(
                    f"Using description from desc.json for key '{key_for_desc}'")
            except Exception as desc_error:
                print(f"Error using desc.json: {desc_error}")
                image_description = default_description

        # Rest of the function remains unchanged
        openai_edit_prompt = f"""
        Transform into a detailed Manhwa-style digital illustration in the {style_for_prompts}, but keep their original face and ethnicity as Indian only. Exact Resembling face and ethnicity. Keep strong facial resemblance avoid realism but preserve identity. Focus on face with clear open eyes (Spectacles, if visible), accurate facial features, defined shadows. Keep strong facial resemblance avoid realism but preserve identity. Show the character from head down to the knees, and the body should be in a pose that is suitable for the expression according to the image description as: {image_description}
        """

        # Use OpenAI client for image editing
        key = get_next_openai_key()
        if not key:
            raise ValueError("No OpenAI API keys configured")

        client = OpenAI(api_key=key)

        try:
            response = client.images.edit(
                model="gpt-image-1",
                image=open(input_path, "rb"),
                n=1,
                size="1024x1536",
                quality="high",
                prompt=openai_edit_prompt
            )

            image_base64 = response.data[0].b64_json

            # Clean up temp files
            try:
                os.unlink(mask_path)
                os.unlink(input_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp files: {e}")

            return image_base64

        except Exception as e:
            print(
                f"Error in OpenAI image editing: {e}\n{traceback.format_exc()}")
            raise

    except Exception as e:
        print(f"Error in Gemini processing: {e}\n{traceback.format_exc()}")
        raise
