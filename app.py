import os
import tempfile
from flask import Flask, request, jsonify
from pathlib import Path
import traceback
import asyncio
from predict import Predictor
from queue_manager import QueueManager  # Import the new manager

# --- Define and ensure temporary directory ---
TEMP_DIR = "./tmp_job_files"  # Relative path within your project
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Ensured temporary directory exists: {os.path.abspath(TEMP_DIR)}")
except OSError as e:
    print(f"FATAL: Could not create temporary directory {TEMP_DIR}: {e}")
    # Depending on requirements, you might want to exit or raise here
    # For now, we'll let it proceed and potentially fail later if creation failed.

app = Flask(__name__)

# --- Initialization ---
print("Initializing Predictor...")
predictor = Predictor()
predictor.setup()  # Assuming setup is synchronous and relatively fast
print("Initializing QueueManager...")
# Pass the predictor instance to the manager
queue_manager = QueueManager(predictor)
print("Initialization complete. Starting Flask app...")

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

# Make generate async


@app.route('/generate', methods=['POST'])
async def generate():
    subject_temp_file_path = None
    mask_temp_file_path = None

    try:
        # --- Get Inputs and Save Temporarily ---
        expression = request.form.get(
            'expression', 'happy')  # Default defined here

        if 'subject' not in request.files or not request.files['subject'].filename:
            return jsonify({'error': 'Missing "subject" image file.'}), 400
        subject_file = request.files['subject']
        subject_suffix = Path(subject_file.filename).suffix or '.png'
        # Save with delete=False, manager will clean up
        subject_temp_fd, subject_temp_file_path = tempfile.mkstemp(
            suffix=subject_suffix, dir=TEMP_DIR)
        subject_file.save(subject_temp_file_path)
        os.close(subject_temp_fd)  # Close descriptor after saving
        print(f"Saved subject temp file: {subject_temp_file_path}")

        if 'mask' not in request.files or not request.files['mask'].filename:
            # Clean up subject if mask is missing (before job submission)
            # Use a direct os.unlink here as the job hasn't been submitted yet
            if subject_temp_file_path and os.path.exists(subject_temp_file_path):
                try:
                    os.unlink(subject_temp_file_path)
                    print(
                        f"Cleaned up orphaned subject file: {subject_temp_file_path}")
                except Exception as clean_e:
                    print(
                        f"Warning: Failed to cleanup orphaned subject file {subject_temp_file_path}: {clean_e}")
            # await queue_manager._cleanup_temp_files('generate_endpoint_error', subject_temp_file_path) # Removed
            return jsonify({'error': 'Missing "mask" image file.'}), 400
        mask_file = request.files['mask']
        mask_suffix = Path(mask_file.filename).suffix or '.png'
        # Use the defined TEMP_DIR
        mask_temp_fd, mask_temp_file_path = tempfile.mkstemp(
            suffix=mask_suffix, dir=TEMP_DIR)
        mask_file.save(mask_temp_file_path)
        os.close(mask_temp_fd)
        print(f"Saved mask temp file: {mask_temp_file_path}")

        # --- Prepare Job Data ---
        # Use defaults from predict.py or allow overrides from form data
        try:
            job_data = {
                "input_image_path": subject_temp_file_path,
                "mask_image_path": mask_temp_file_path,
                "expression": expression,
                "seed": int(request.form.get('seed', 42)),
                "height": int(request.form.get('height', 768)),
                "width": int(request.form.get('width', 512)),
                "subject_lora_scale": float(request.form.get('subject_lora_scale', 1.0)),
                "inpainting_lora_scale": float(request.form.get('inpainting_lora_scale', 1.0))
            }
        except ValueError as e:
            # Handle potential errors converting form data to int/float
            # Clean up both files manually if job data prep fails
            if subject_temp_file_path and os.path.exists(subject_temp_file_path):
                try:
                    os.unlink(subject_temp_file_path)
                except Exception:
                    pass  # Ignore cleanup error
            if mask_temp_file_path and os.path.exists(mask_temp_file_path):
                try:
                    os.unlink(mask_temp_file_path)
                except Exception:
                    pass  # Ignore cleanup error
            # await queue_manager._cleanup_temp_files('generate_endpoint_error', subject_temp_file_path, mask_temp_file_path) # Removed
            return jsonify({'error': f'Invalid parameter format: {e}'}), 400

        # --- Submit Job to Queue Manager ---
        # Don't log full temp paths here for brevity/security
        print(
            f"Submitting job: expression={job_data['expression']}, seed={job_data['seed']}, size={job_data['width']}x{job_data['height']}")
        job_id = await queue_manager.submit_job(job_data)

        # --- Return Job ID Immediately ---
        # Cleanup is now handled by the QueueManager's processing functions
        # 202 Accepted
        return jsonify({'job_id': job_id, 'status': 'submitted'}), 202

    except Exception as e:
        print(f"Error in /generate endpoint: {e}\n{traceback.format_exc()}")
        # Critical error during submission *after* file creation but *before* job acceptance.
        # The QueueManager won't know about these files. Attempt manual cleanup.
        # This is a fallback; ideally QueueManager handles all cleanup post-submission.
        if subject_temp_file_path and os.path.exists(subject_temp_file_path):
            try:
                os.unlink(subject_temp_file_path)
            except Exception:
                print(
                    f"Warning: Failed final cleanup for {subject_temp_file_path}")
        if mask_temp_file_path and os.path.exists(mask_temp_file_path):
            try:
                os.unlink(mask_temp_file_path)
            except Exception:
                print(
                    f"Warning: Failed final cleanup for {mask_temp_file_path}")
        # Removed: await queue_manager._cleanup_temp_files(...)
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
