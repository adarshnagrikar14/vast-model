import os
import tempfile
from flask import Flask, request, send_file, jsonify
from predict import Predictor
from pathlib import Path
import traceback  # Import traceback for detailed error logging

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
    output_temp_file = None

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
            expression=expression, seed=25, height  = 768, width = 512, subject_lora_scale=1.1, inpainting_lora_scale=1.18
        )
        print(f"Predictor returned output path: {output_path_predictor}")

        # --- Prepare Response ---
        # Create a separate temporary file for sending the response to avoid issues
        # if the original predictor output needs to persist or gets cleaned up elsewhere.
        output_temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.png')
        with open(output_path_predictor, 'rb') as f_in:
            with open(output_temp_file.name, 'wb') as f_out:
                f_out.write(f_in.read())
        print(
            f"Copied predictor output to response temp file: {output_temp_file.name}")

        response = send_file(output_temp_file.name, mimetype='image/png')

        # --- Cleanup ---
        # Use call_on_close for robust cleanup even if client disconnects
        @response.call_on_close
        def cleanup():
            print("Cleaning up temporary files...")
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
            # Clean up the predictor's original output file if it's different from the response temp file
            # This might be the same file if predict.py saves directly to OUTPUT_FILENAME
            # Assuming predict.py saves here
            predictor_output_filename = Path("output.png")
            if os.path.exists(predictor_output_filename) and \
               (not output_temp_file or Path(output_temp_file.name).resolve() != predictor_output_filename.resolve()):
                try:
                    os.unlink(predictor_output_filename)
                    print(
                        f"Deleted predictor output file: {predictor_output_filename}")
                except Exception as e:
                    print(
                        f"Error deleting predictor output file {predictor_output_filename}: {e}")

            # Clean up the response temporary file last
            if output_temp_file and os.path.exists(output_temp_file.name):
                try:
                    os.unlink(output_temp_file.name)
                    print(
                        f"Deleted response temp file: {output_temp_file.name}")
                except Exception as e:
                    print(
                        f"Error deleting response temp file {output_temp_file.name}: {e}")

        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())  # Print full traceback for debugging
        # Ensure cleanup happens even on error before returning response
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
        if output_temp_file and os.path.exists(output_temp_file.name):
            try:
                os.unlink(output_temp_file.name)
                print(
                    f"Cleaned up output temp file on error: {output_temp_file.name}")
            except Exception as clean_e:
                print(f"Error during error cleanup (output): {clean_e}")

        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
