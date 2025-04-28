import os
import torch
import requests
import io
from dotenv import load_dotenv
from PIL import Image as PILImage
from huggingface_hub import login
import google.generativeai as genai
from src.pipeline import FluxPipeline
from cog import BasePredictor, Input, Path
from src.lora_helper import set_single_lora
from src.transformer_flux import FluxTransformer2DModel

import sys
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

load_dotenv()
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

gemini_api_key = os.environ.get("GEMINI_KEY")

BASE_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
LORA_BASE_PATH = "./models"
OUTPUT_FILENAME = "output.png"

SYSTEM_PROMPT = """
You are an expert image analyzer specializing in generating highly detailed, descriptive text prompts for AI image generation models, particularly for character creation. Your task is to analyze the provided image and create a text prompt broken down into the following sections, using the exact labels provided:

1.  **Gender Description:** Clearly depict the subject's perceived gender. Describe typically masculine or feminine features relevant to the depiction (e.g., broader shoulders, sharper facial features for male; softer jawline, narrower shoulders for female). Mention estimated build or frame if apparent.

2.  **Face Structure:** Describe the subject's facial features with rich detail and evocative adjectives, prioritizing accurate representation of the original face structure. Maintain the subject's unique features without stylization. Include:
    *   Overall face shape (e.g., square, oval, round etc).
    *   Jawline (e.g., sharp, defined, soft etc).
    *   Cheekbones (e.g., high, prominent, subtle etc).
    *   Eyes (e.g., color, size, shape, expression, details like 'sparkling', 'intricate shading', 'long eyelashes' etc).
    *	If a girl, any make-up (e.g lipstick shade, eyeliner)
    *   Nose (e.g., size, shape etc).
    * 	Teeths if visible
    *   Lips (e.g., full, thin, shape etc).
    *   Eyebrows (e.g., thick, thin, expressive, shape etc).
    *   Skin tone (e.g., warm medium, fair, deep) and texture (e.g., smooth, clear, subtle wrinkles etc).
    *   Estimated age range (e.g., late 20s, early 40s etc).
    *   Distinctive features (e.g., beard, mustache, dimples, scars, glasses etc). Describe facial hair with detail (length, texture, grooming). Note absence of features if relevant (e.g., no glasses, minimal wrinkles).

3.  **Hair:** Describe the subject's hair comprehensively:
    *   Color (e.g., dark brown, blonde, black with vibrant highlights etc).
    *   Length (e.g., short, medium, long etc).
    *   Style (e.g., styled with dynamic flow, voluminous texture, spiked upwards, natural waves, straight etc).
    *   Texture (e.g., slight sheen, glossy, wavy, curly, straight etc).
    *   Notable features (e.g., highlights, receding hairline, bangs etc).

Format the output using the labels "Gender Description:", "Face Structure:", "Hair:" clearly separating each section. Use descriptive language suitable for an AI image generator. Focus *only* on these three areas. Do not create any image of your own.
"""


def analyze_image(image_input, system_prompt: str, model):
    """
    Analyzes an image using the provided Gemini model.

    Args:
        image_input: Can be a URL string, a Path object, or a PIL Image object
        system_prompt: The system prompt to guide Gemini's analysis
        model: The configured Gemini model

    Returns:
        The generated text or None on error.
    """
    if not model:
        print("Error: Gemini model not configured.")
        return None

    try:
        img = None

        # Handle different input types
        if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
            # It's a URL
            response = requests.get(image_input, stream=True)
            response.raise_for_status()
            img = PILImage.open(io.BytesIO(response.content))
        elif isinstance(image_input, (str, Path)):
            # It's a file path
            img = PILImage.open(str(image_input))
        elif isinstance(image_input, PILImage.Image):
            # It's already a PIL Image
            img = image_input
        else:
            print(f"Error: Unsupported image input type: {type(image_input)}")
            return None

        # Convert image to bytes
        buffered = io.BytesIO()
        img_format = "JPEG"  # Default format
        if hasattr(img, 'format') and img.format:
            img_format = img.format

        img.save(buffered, format=img_format)
        image_bytes = buffered.getvalue()

        # Prepare for Gemini
        image_parts = [
            {"mime_type": f"image/{img_format.lower()}", "data": image_bytes}]
        prompt_parts = [system_prompt, image_parts[0]]

        # Generate content
        gen_response = model.generate_content(prompt_parts, stream=False)

        # Safety check
        if gen_response.prompt_feedback and gen_response.prompt_feedback.block_reason:
            print(
                f"Content blocked due to {gen_response.prompt_feedback.block_reason}")
            return None

        if not gen_response.text:
            print("Error: No text generated by Gemini.")
            return None

        return gen_response.text

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from URL: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during Gemini analysis: {e}")
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

        # Configure Gemini API
        self.gemini_model = None
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel(
                    'gemini-2.0-flash')
                print("Gemini API configured successfully with gemini-2.0-flash.")
            except Exception as e:
                print(f"Warning: Failed to configure Gemini API: {e}")
        else:
            print("Warning: GEMINI_KEY environment variable not set. Prompt generation from image URL will be disabled.")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt (used if Gemini image analysis is not enabled)"),
        spatial_img: Path = Input(
            description="Spatial control image (also used for Gemini analysis if available)", default=None),
        base_prompt: str = Input(
            description="Text to prepend to the Gemini-generated prompt (optional)",
            default="""
            Create a K-pop manhwa style digital illustration of this image. Transform the person into a detailed manhwa-style artwork with the following specifications:
            Art Style: Resemble K-pop manhwa style with clean lines, vibrant colors, and exaggerated features.
            Outfit: Gender-specific outfit, modern K-pop-inspired attire: a tailored jacket/suit/coat featuring intricate patterns and a belted waist. Add glowing or magical elements for a mystical vibe.
            Background: Fiery with deep red tones, crackling energy, glowing sparks, and lightning effects for a dramatic, mystical atmosphere.
            Additional Elements: Add a magical aura with glowing energy around hands, small star-like symbols in the background, and subtle K-pop-inspired accessories.
            Pose and Expression: Match the pose with cosmic fire in hand, glowing, enhance with manhwa-style drama—confident and serene expression.
            """
        ),
        height: int = Input(
            description="Height of the output image", default=768),
        width: int = Input(
            description="Width of the output image", default=512),
        seed: int = Input(description="Random seed", default=16),
        control_type: str = Input(
            description="LoRA control type", default="Manhwa", choices=["Manhwa", "None"]),
        lora_scale: float = Input(
            description="Scale for the LoRA weights", default=0.85, ge=0.0, le=2.0)
    ) -> Path:

        # --- Prompt Generation Logic ---
        final_prompt = prompt
        # Only try Gemini analysis if we have both a spatial image and the model is configured
        if spatial_img and self.gemini_model:
            print(
                f"Attempting to generate prompt from spatial image: {spatial_img}")
            generated_desc = analyze_image(
                spatial_img, SYSTEM_PROMPT, self.gemini_model
            )
            if generated_desc:
                formatted_desc = generated_desc.replace("**", "")
                final_prompt = f"{base_prompt} {formatted_desc}".strip()
                print("Using Gemini-generated prompt:")
                print(final_prompt)
            else:
                print(
                    "Warning: Gemini prompt generation failed. Falling back to standard prompt.")
                final_prompt = prompt
        else:
            if spatial_img and not self.gemini_model:
                print(
                    "Warning: Spatial image provided, but Gemini model is not configured. Using standard prompt.")
            final_prompt = prompt

        if control_type == "Manhwa":
            lora_path = os.path.join(LORA_BASE_PATH, "studio.safetensors")
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA file not found at {lora_path}")
            set_single_lora(self.pipe.transformer, lora_path,
                            lora_weights=[lora_scale], cond_size=512)

        spatial_imgs_pil = []
        if spatial_img:
            try:
                pil_image = PILImage.open(str(spatial_img))
                spatial_imgs_pil = [pil_image]
            except Exception as e:
                print(f"Warning: Could not process spatial image: {e}")

        generator = torch.Generator(self.device).manual_seed(seed)

        generated_pil_image = self.pipe(
            prompt=final_prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=25,
            max_sequence_length=512,
            generator=generator,
            subject_images=[],
            spatial_images=spatial_imgs_pil,
            cond_size=512,
        ).images[0]

        output_path = Path(OUTPUT_FILENAME)
        generated_pil_image.save(output_path)

        return output_path
