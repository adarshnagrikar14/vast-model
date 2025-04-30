import os
import tempfile
from flask import Flask, request, send_file, jsonify
from predict import Predictor
from pathlib import Path
import traceback  # Import traceback for detailed error logging
import base64  # Add base64 import

app = Flask(__name__)

# Initialize the predictor
predictor = Predictor()
predictor.setup()


@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "endpoint": "/generate",
        "method": "POST",
        "description": "Generate image using form data based on subject, mask, and expression.",
        "parameters": {
            "subject": "Input image file for subject conditioning.",
            "mask": "Inpainting mask image file (white areas will be inpainted).",
            "expression": "Desired facial expression for the manhwa style (e.g., 'happy', 'angry', 'surprised', 'neutral', default: 'angry')."
        },
        "example_curl": 'curl -X POST http://localhost:8000/generate -F "subject=@/path/to/subject.png" -F "mask=@/path/to/mask.png" -F "expression=happy" --output result.png'
    })


@app.route('/generate', methods=['POST'])
def generate():
    subject_temp_file = None
    mask_temp_file = None
    # output_temp_file = None # No longer needed for response, only for predictor output handling

    try:
        # --- Get Inputs ---
        # Expression
        # Default to 'angry' if not provided
        expression = request.form.get('expression', 'angry')

        # Subject Image
        if 'subject' not in request.files or not request.files['subject'].filename:
            return jsonify({'error': 'Missing "subject" image file in form data.'}), 400
        subject_file = request.files['subject']
        # Keep original extension if possible
        subject_suffix = Path(subject_file.filename).suffix or '.png'
        subject_temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=subject_suffix)
        subject_file.save(subject_temp_file.name)
        subject_path = Path(subject_temp_file.name)
        print(f"Saved subject image to temporary file: {subject_path}")

        # Mask Image
        if 'mask' not in request.files or not request.files['mask'].filename:
            # Clean up subject temp file if mask is missing
            os.unlink(subject_path)
            return jsonify({'error': 'Missing "mask" image file in form data.'}), 400
        mask_file = request.files['mask']
        # Keep original extension
        mask_suffix = Path(mask_file.filename).suffix or '.png'
        mask_temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=mask_suffix)
        mask_file.save(mask_temp_file.name)
        mask_path = Path(mask_temp_file.name)
        print(f"Saved mask image to temporary file: {mask_path}")

        # --- Call Predictor ---
        # Uses defaults from predict.py for height, width, seed, lora_scales
        print(
            f"Calling predictor with subject: {subject_path}, mask: {mask_path}, expression: {expression}")
        output_path_predictor = predictor.predict(
            input_image=subject_path,
            inpainting_mask=mask_path,
            expression=expression, seed=25, height=768, width=512, subject_lora_scale=1.1, inpainting_lora_scale=1.18
        )
        print(f"Predictor returned output path: {output_path_predictor}")

        # --- Prepare Response ---
        # Read the output file and encode it as Base64
        image_base64 = None
        try:
            with open(output_path_predictor, 'rb') as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                print(f"Encoded image from {output_path_predictor} to Base64.")
        except Exception as e:
            print(f"Error reading or encoding predictor output file: {e}")
            # Clean up input files before raising internal error
            if subject_temp_file and os.path.exists(subject_temp_file.name):
                os.unlink(subject_temp_file.name)
            if mask_temp_file and os.path.exists(mask_temp_file.name):
                os.unlink(mask_temp_file.name)
            raise  # Re-raise the exception to be caught by the outer try-except

        # Prepare the JSON response
        response_data = {'image_base64': image_base64}

        # --- Cleanup ---
        # Use a function to schedule cleanup *after* the request context ends
        # This is simpler than call_on_close when not using send_file
        def cleanup_files():
            print("Cleaning up temporary files...")
            # Input files
            if subject_temp_file and os.path.exists(subject_temp_file.name):
                try:
                    os.unlink(subject_temp_file.name)
                    print(
                        f"Deleted subject temp file: {subject_temp_file.name}")
                except Exception as e:
                    print(
                        f"Error deleting subject temp file {subject_temp_file.name}: {e}")
            if mask_temp_file and os.path.exists(mask_temp_file.name):
                try:
                    os.unlink(mask_temp_file.name)
                    print(f"Deleted mask temp file: {mask_temp_file.name}")
                except Exception as e:
                    print(
                        f"Error deleting mask temp file {mask_temp_file.name}: {e}")

            # Predictor output file (make sure it exists before deleting)
            # Use the actual path returned
            predictor_output_path = Path(output_path_predictor)
            if predictor_output_path.exists():
                try:
                    os.unlink(predictor_output_path)
                    print(
                        f"Deleted predictor output file: {predictor_output_path}")
                except Exception as e:
                    print(
                        f"Error deleting predictor output file {predictor_output_path}: {e}")

        # Register cleanup function to run after request is complete
        from flask import after_this_request

        @after_this_request
        def schedule_cleanup(response):
            cleanup_files()
            return response

        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())  # Print full traceback for debugging
        # Ensure cleanup happens even on error before returning response
        # Note: predictor output might not exist or be cleaned up if error was before/during prediction
        if subject_temp_file and os.path.exists(subject_temp_file.name):
            try:
                os.unlink(subject_temp_file.name)
                print(
                    f"Cleaned up subject temp file on error: {subject_temp_file.name}")
            except Exception as clean_e:
                print(f"Error during error cleanup (subject): {clean_e}")
        if mask_temp_file and os.path.exists(mask_temp_file.name):
            try:
                os.unlink(mask_temp_file.name)
                print(
                    f"Cleaned up mask temp file on error: {mask_temp_file.name}")
            except Exception as clean_e:
                print(f"Error during error cleanup (mask): {clean_e}")
        # Don't try to delete output_temp_file as it's removed
        # Predictor output cleanup is handled by the main cleanup logic or might fail if prediction failed

        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
