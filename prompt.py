import google.generativeai as genai
from PIL import Image
import io
import requests  # Import the requests library

# Replace with your actual Gemini API key
GOOGLE_API_KEY = "AIzaSyB4kcWMemFLQ8dkJqgdTsOmN1NN5pwO30o"

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)


def analyze_image_from_url(image_url, system_prompt):
    """
    Analyzes an image from a URL and returns an AI-generated text prompt
    focusing on facial structure, hair, and gender description.

    Args:
        image_url: URL of the image.
        system_prompt: A detailed system prompt to guide the model's output.

    Returns:
        A string containing the generated prompt. Returns None if there's an error.
    """
    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Open the image using PIL
        img = Image.open(io.BytesIO(response.content))

        # Convert image to bytes
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")  # Or PNG, depending on the image
        image_bytes = buffered.getvalue()

        # Prepare the image data for the model
        # Adjust mime type if needed
        image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]

        # Construct the prompt
        prompt_parts = [
            system_prompt,
            image_parts[0]  # Add image part to prompt
        ]

        # Load the Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Send request to model
        response = model.generate_content(prompt_parts, stream=False)

        # Check the response for errors
        if response.prompt_feedback and response.prompt_feedback.safety_ratings:
            for rating in response.prompt_feedback.safety_ratings:
                if rating.blocked:
                    print(
                        f"Content blocked due to {rating.category}: {rating.probability}")
                    return None  # Or raise an exception

        # Check for empty response or errors in the response
        if not response.text:
            print("Error: No text generated.")
            return None

        return response.text

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


SYSTEM_PROMPT = """
You are an expert image analyzer specializing in generating highly detailed, descriptive text prompts for AI image generation models, particularly for character creation. Your task is to analyze the provided image and create a text prompt broken down into the following sections, using the exact labels provided:

1.  **Gender Description:** Clearly depict the subject's perceived gender. Describe typically masculine or feminine features relevant to the depiction (e.g., broader shoulders, sharper facial features for male; softer jawline, narrower shoulders for female). Mention estimated build or frame if apparent.

2.  **Face Structure:** Describe the subject's facial features with rich detail and evocative adjectives. Include:
    *   Overall face shape (e.g., square, oval, round).
    *   Jawline (e.g., sharp, defined, soft).
    *   Cheekbones (e.g., high, prominent, subtle).
    *   Eyes (e.g., color, size, shape, expression, details like 'sparkling', 'intricate shading', 'long eyelashes').
    *   Nose (e.g., size, shape).
    *   Lips (e.g., full, thin, shape).
    *   Eyebrows (e.g., thick, thin, expressive, shape).
    *   Skin tone (e.g., warm medium, fair, deep) and texture (e.g., smooth, clear, subtle wrinkles).
    *   Estimated age range (e.g., late 20s, early 40s).
    *   Distinctive features (e.g., beard, mustache, dimples, scars, glasses). Describe facial hair with detail (length, texture, grooming). Note absence of features if relevant (e.g., no glasses, minimal wrinkles).

3.  **Hair:** Describe the subject's hair comprehensively:
    *   Color (e.g., dark brown, blonde, black with vibrant highlights).
    *   Length (e.g., short, medium, long).
    *   Style (e.g., styled with dynamic flow, voluminous texture, spiked upwards, natural waves, straight). Use specific style references if applicable (e.g., K-pop-inspired).
    *   Texture (e.g., slight sheen, glossy, wavy, curly, straight).
    *   Notable features (e.g., highlights, receding hairline, bangs).

Format the output using the labels "Gender Description:", "Face Structure:", "Hair:" clearly separating each section. Use descriptive language suitable for an AI image generator. Focus *only* on these three areas. Do not create any image of your own.
"""

IMAGE_URL = "https://hips.hearstapps.com/hmg-prod/images/mh-6-7-bald-celebs-1623156958.png?crop=0.502xw:1.00xh;0,0&resize=640:*"

generated_prompt = analyze_image_from_url(IMAGE_URL, SYSTEM_PROMPT)

if generated_prompt:
    print("Generated Prompt:\n", generated_prompt)
else:
    print("Failed to generate prompt.")
