import os
import io
import sys
import torch
import tempfile
import traceback
import base64
import replicate
import requests
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image as PILImage
from huggingface_hub import login
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
DEFAULT_SUBJECT_LORA_SCALE = 1.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_INPAINTING_LORA_SCALE = 1.0

try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except OSError as e:
    print(f"FATAL: Could not create temporary directory {TEMP_DIR}: {e}")


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


def predict_replicate(
    input_image_bytes: bytes,
    mask_image_bytes: bytes,
    expression: str = "k-pop happy",
) -> str | None:
    output_path_str = None
    try:
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print(
                "Warning: REPLICATE_API_TOKEN environment variable not set. Replicate fallback disabled.")
            return None

        input_image_uri = image_bytes_to_data_uri(input_image_bytes)
        mask_image_uri = image_bytes_to_data_uri(mask_image_bytes)

        print("Calling Replicate API...")
        output_url = replicate.run(
            "adarshnagrikar14/manhwa-ai:0ed8ac8e28cfb050730eb3e1fbcbc9c60d7001e3a53931cc4f3c44cf08bab659",
            input={
                "seed": DEFAULT_SEED,
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "expression": expression,
                "subject_lora_scale": DEFAULT_SUBJECT_LORA_SCALE,
                "inpainting_lora_scale": DEFAULT_INPAINTING_LORA_SCALE,
                "input_image": input_image_uri,
                "inpainting_mask": mask_image_uri
            }
        )

        if not output_url or not isinstance(output_url, str):
            raise ValueError(
                f"Invalid output received from Replicate: {output_url}")

        response = requests.get(output_url, stream=True)
        response.raise_for_status()

        output_fd, output_path_str = tempfile.mkstemp(
            suffix=".png", prefix="replicate_output_", dir=TEMP_DIR)
        with os.fdopen(output_fd, 'wb') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)

        print(f"Replicate output saved to: {output_path_str}")
        return output_path_str

    except replicate.exceptions.ReplicateError as e:
        print(
            f"ERROR during Replicate prediction: {e}\n{traceback.format_exc()}")
        if output_path_str and os.path.exists(output_path_str):
            try:
                os.unlink(output_path_str)
            except Exception:
                pass
        raise
    except requests.exceptions.RequestException as e:
        print(
            f"ERROR downloading image from Replicate: {e}\n{traceback.format_exc()}")
        if output_path_str and os.path.exists(output_path_str):
            try:
                os.unlink(output_path_str)
            except Exception:
                pass
        return None
    except Exception as e:
        print(
            f"ERROR during Replicate fallback function: {e}\n{traceback.format_exc()}")
        if output_path_str and os.path.exists(output_path_str):
            try:
                os.unlink(output_path_str)
            except Exception:
                pass
        raise
        return None


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
            Crop the face precisely and convert it into a digital illustration in {fixed_expression} style.
            Maintain exact hair style and keep eyes open. Pose with a slight rightward turn.
            Slightly widen the face while preserving original structure and strong likeness.
            Retain fine facial details—including lines and wrinkles (if present).
            For K-pop style, apply smooth skin, stylized features, and expressive eyes while maintaining resemblance.
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

            flux_prompt = "put exact face on the body, match body skin tone"
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
            unset_lora(self.pipe.transformer)
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
