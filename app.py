import os
import tempfile
from flask import Flask, request, jsonify
from pathlib import Path
import traceback
import asyncio
import io
import base64
from PIL import Image as PILImage
from openai import OpenAI
from predict import Predictor
from queue_manager import QueueManager

# --- Define and ensure temporary directory ---
TEMP_DIR = "./tmp_job_files"
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Ensured temporary directory exists: {os.path.abspath(TEMP_DIR)}")
except OSError as e:
    print(f"FATAL: Could not create temporary directory {TEMP_DIR}: {e}")

app = Flask(__name__)

# --- Initialization ---
print("Initializing Predictor...")
predictor = Predictor()
predictor.setup()
print("Initializing QueueManager...")
queue_manager = QueueManager(predictor)

# --- OpenAI Client Setup ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        print("OpenAI client configured successfully.")
    except Exception as e:
        print(
            f"Warning: Failed to configure OpenAI client: {e}. OpenAI editing step may fail.")
else:
    print("Warning: OPENAI_API_KEY environment variable not set. OpenAI editing step will be skipped.")

print("Initialization complete. Starting Flask app...")

# --- OpenAI Edit Function (Copied from original predict.py) ---


def edit_image_openai(client: OpenAI, input_image_path: str, edit_prompt: str) -> PILImage.Image | None:
    """Uses OpenAI's image edit API, handling b64_json response."""
    if not client:
        print("Skipping OpenAI edit: Client not configured.")
        # Return None to indicate skipping, let caller handle original image
        return None
    try:
        print(f"Sending image '{input_image_path}' to OpenAI for editing...")
        with open(input_image_path, "rb") as img_file:
            response = client.images.edit(
                model="dall-e-2",  # Or gpt-image-1 if that was intended - DALL-E 2 edit endpoint is standard
                image=img_file,
                prompt=edit_prompt,
                n=1,
                size="1024x1024",  # Ensure this size is supported
                response_format="b64_json"  # Request b64_json explicitly
            )

        b64_data = response.data[0].b64_json
        if not b64_data:
            print("Error: OpenAI response did not contain b64_json data.")
            return None

        print("Received b64_json data from OpenAI. Decoding...")
        image_bytes = base64.b64decode(b64_data)
        edited_image_pil = PILImage.open(
            io.BytesIO(image_bytes)).convert("RGB")
        print("Successfully created PIL image from OpenAI b64_json response.")
        return edited_image_pil

    except FileNotFoundError:
        print(
            f"Error during OpenAI call: Input image file not found at {input_image_path}")
        return None
    except AttributeError:
        print(
            "Error during OpenAI call: Could not find 'b64_json' in OpenAI response data.")
        return None
    except base64.binascii.Error as b64_error:
        print(f"Error decoding Base64 data from OpenAI: {b64_error}")
        return None
    except Exception as e:
        print(
            f"An error occurred during OpenAI image editing: {e}\n{traceback.format_exc()}")
        return None

# --- Routes ---


@app.route('/')
def index():
    # Simplified index for clarity
    return jsonify({
        "status": "online",
        "message": "Manhwa AI Generation Service",
        "endpoints": {
            "submit_job": {
                "path": "/generate",
                "method": "POST",
                "description": "Submit image generation job (form data: subject, mask, expression, [optional params])",
                "response": "{'job_id': '...', 'status': 'submitted'}"
            },
            "check_status": {
                "path": "/queue/<job_id>",
                "method": "GET",
                "description": "Check job status and get result",
                "response": "{'job_id': '...', 'status': 'pending|processing_local|processing_replicate|completed|failed', ['image_base64': '...', 'error_message': '...']}"
            }
        }
    })


@app.route('/generate', methods=['POST'])
async def generate():
    subject_temp_file_path = None
    mask_temp_file_path = None
    subject_bytes = None
    mask_bytes = None
    openai_edited_temp_path = None  # Path for edited image if saved

    try:
        expression = request.form.get('expression', 'happy')

        # --- Handle Subject Image (Save Temporarily for OpenAI) ---
        if 'subject' not in request.files or not request.files['subject'].filename:
            return jsonify({'error': 'Missing "subject" image file.'}), 400
        subject_file = request.files['subject']
        subject_suffix = Path(subject_file.filename).suffix or '.png'
        subject_filename = subject_file.filename

        subject_temp_fd, subject_temp_file_path = tempfile.mkstemp(
            suffix=subject_suffix, dir=TEMP_DIR)
        subject_file.save(subject_temp_file_path)
        os.close(subject_temp_fd)
        print(
            f"Saved original subject '{subject_filename}' to temp file: {subject_temp_file_path}")

        # --- OpenAI Editing Step ---
        openai_edit_prompt = f"""
        Crop the face precisely and convert it into a digital illustration in {expression} style.
        Maintain exact hair style and keep eyes open. Pose with a slight rightward turn.
        Slightly widen the face while preserving original structure and strong likeness.
        Retain fine facial details—including lines and wrinkles (if present).
        For K-pop style, apply smooth skin, stylized features, and expressive eyes while maintaining resemblance.
        """
        print(
            f"Constructed OpenAI Edit Prompt (Expression: {expression})")  # Simplified log

        edited_subject_pil = edit_image_openai(
            openai_client, subject_temp_file_path, openai_edit_prompt)

        # Load final subject bytes (either edited or original)
        if edited_subject_pil:
            print("OpenAI edit successful. Converting edited PIL to bytes.")
            # Convert edited PIL image to bytes
            img_byte_arr = io.BytesIO()
            edited_subject_pil.save(
                img_byte_arr, format='PNG')  # Save as PNG bytes
            subject_bytes = img_byte_arr.getvalue()
            print(f"Converted edited image to {len(subject_bytes)} bytes.")
            # Optionally save the edited image temporarily if needed elsewhere, remember to clean up
            # edit_fd, openai_edited_temp_path = tempfile.mkstemp(suffix=".png", dir=TEMP_DIR)
            # edited_subject_pil.save(openai_edited_temp_path)
            # os.close(edit_fd)
        else:
            print("OpenAI edit skipped or failed. Using original subject image bytes.")
            # Read the original image bytes if OpenAI failed/skipped
            with open(subject_temp_file_path, 'rb') as f:
                subject_bytes = f.read()
            print(
                f"Read original subject file ({len(subject_bytes)} bytes) into memory.")

        # Clean up the original subject temp file now
        if subject_temp_file_path and os.path.exists(subject_temp_file_path):
            try:
                os.unlink(subject_temp_file_path)
                print(
                    f"Deleted original subject temp file: {subject_temp_file_path}")
            except Exception as e:
                print(
                    f"Warning: Failed to delete original subject temp file {subject_temp_file_path}: {e}")

        # --- Handle Mask Image (Read Bytes directly) ---
        if 'mask' not in request.files or not request.files['mask'].filename:
            # Clean up any potential temp edited file if created
            if openai_edited_temp_path and os.path.exists(openai_edited_temp_path):
                try:
                    os.unlink(openai_edited_temp_path)
                except Exception:
                    pass
            return jsonify({'error': 'Missing "mask" image file.'}), 400
        mask_file = request.files['mask']
        mask_filename = mask_file.filename
        # Read mask directly into bytes - no need to save unless OpenAI needs it too
        mask_bytes = mask_file.read()
        print(
            f"Read mask file '{mask_filename}' ({len(mask_bytes)} bytes) directly into memory.")
        # mask_suffix = Path(mask_file.filename).suffix or '.png'
        # mask_temp_fd, mask_temp_file_path = tempfile.mkstemp(suffix=mask_suffix, dir=TEMP_DIR)
        # try:
        #     mask_file.save(mask_temp_file_path)
        #     os.close(mask_temp_fd)
        #     with open(mask_temp_file_path, 'rb') as f:
        #         mask_bytes = f.read()
        #     print(f"Read mask file '{mask_filename}' ({len(mask_bytes)} bytes) into memory.")
        # finally:
        #     if mask_temp_file_path and os.path.exists(mask_temp_file_path):
        #         try: os.unlink(mask_temp_file_path)
        #         except Exception as e: print(f"Warning: Failed to delete temp mask file {mask_temp_file_path}: {e}")

        # --- Prepare Job Data ---
        try:
            job_data = {
                # Use final (edited or original) bytes
                "input_image_bytes": subject_bytes,
                "mask_image_bytes": mask_bytes,
                "expression": expression,
                "seed": int(request.form.get('seed', 42)),
                "height": int(request.form.get('height', 768)),
                "width": int(request.form.get('width', 512)),
                "subject_lora_scale": float(request.form.get('subject_lora_scale', 1.0)),
                "inpainting_lora_scale": float(request.form.get('inpainting_lora_scale', 1.0))
            }
        except ValueError as e:
            # Clean up any potential temp edited file if created
            if openai_edited_temp_path and os.path.exists(openai_edited_temp_path):
                try:
                    os.unlink(openai_edited_temp_path)
                except Exception:
                    pass
            return jsonify({'error': f'Invalid parameter format: {e}'}), 400

        # --- Submit Job ---
        print(f"Submitting job with processed subject bytes...")
        job_id = await queue_manager.submit_job(job_data)

        return jsonify({'job_id': job_id, 'status': 'submitted'}), 202

    except Exception as e:
        print(f"Error in /generate endpoint: {e}\n{traceback.format_exc()}")
        # Clean up any temp files that might linger
        if subject_temp_file_path and os.path.exists(subject_temp_file_path):
            try:
                os.unlink(subject_temp_file_path)
            except Exception:
                pass
        if mask_temp_file_path and os.path.exists(mask_temp_file_path):
            try:
                os.unlink(mask_temp_file_path)
            except Exception:
                pass
        if openai_edited_temp_path and os.path.exists(openai_edited_temp_path):
            try:
                os.unlink(openai_edited_temp_path)
            except Exception:
                pass
        return jsonify({'error': 'An internal error occurred during job submission.', 'detail': str(e)}), 500

# New async endpoint to check job status


@app.route('/queue/<job_id>', methods=['GET'])
async def get_status(job_id):
    print(f"Checking status for job_id: {job_id}")
    status_info = await queue_manager.get_job_status(job_id)

    if status_info is None:
        return jsonify({'error': 'Job ID not found'}), 404

    response_data = {
        'job_id': job_id,
        'status': status_info['status']
    }

    if status_info['status'] == 'completed':
        response_data['image_base64'] = status_info['result']
        return jsonify(response_data), 200
    elif status_info['status'] == 'failed':
        response_data['error_message'] = status_info.get(
            'result', 'Unknown error')  # Get error message if available
        # Return 200 OK but indicate failure in status
        return jsonify(response_data), 200
    else:
        # Status is pending, processing_local, or processing_replicate
        # Still processing, return current status
        return jsonify(response_data), 200

# Running the app requires an ASGI server
# Example: pip install uvicorn
# Run: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
# Note: --workers 1 is important for shared resources like the predictor/GPU unless managed carefully.
if __name__ == '__main__':
    print("Starting application using Flask development server (NOT recommended for async).")
    print("Please use an ASGI server like Uvicorn or Hypercorn for production/proper async handling:")
    print("Example: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1")
    # Running with app.run() will likely lead to issues with asyncio concurrency.
    # Disable debug for clarity, set False
    app.run(host='0.0.0.0', port=8000, debug=False)
