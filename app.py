import os
import base64
import traceback
from predict import Predictor
from flask import Flask, request, jsonify

TEMP_DIR = "./tmp_job_files"
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except OSError as e:
    print(f"FATAL: Could not create temporary directory {TEMP_DIR}: {e}")

app = Flask(__name__)

predictor = Predictor()
predictor.setup()


@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "message": "Manhwa AI Generation Service",
        "endpoints": {
            "generate": {
                "path": "/generate",
                "method": "POST",
                "description": "Submit image generation job (form data: subject, mask). Returns result directly.",
                "response": "{'image_base64': '...' | 'error': '...'}"
            }
        }
    })


@app.route('/generate', methods=['POST'])
def generate():
    subject_bytes = None
    mask_bytes = None
    output_image_path = None
    expression = "k-pop happy"

    try:
        if 'subject' not in request.files or not request.files['subject'].filename:
            return jsonify({'error': 'Missing "subject" image file.'}), 400
        subject_file = request.files['subject']
        subject_bytes = subject_file.read()

        if 'mask' not in request.files or not request.files['mask'].filename:
            return jsonify({'error': 'Missing "mask" image file.'}), 400
        mask_file = request.files['mask']
        mask_bytes = mask_file.read()

        if 'expression' in request.form:
            expression = request.form['expression']

        output_image_path = predictor.predict(
            input_image_bytes=subject_bytes,
            mask_image_bytes=mask_bytes,
            expression=expression
        )

        if output_image_path and os.path.exists(output_image_path):
            with open(output_image_path, "rb") as img_file:
                image_base64 = base64.b64encode(
                    img_file.read()).decode('utf-8')
            try:
                os.unlink(output_image_path)
            except Exception as e:
                print(
                    f"Warning: Failed to delete temp output file {output_image_path}: {e}")
            return jsonify({'image_base64': image_base64}), 200
        else:
            return jsonify({'error': 'Prediction finished but output file not found.'}), 500

    except Exception as e:
        print(f"Error in /generate endpoint: {e}\n{traceback.format_exc()}")
        if output_image_path and os.path.exists(output_image_path):
            try:
                os.unlink(output_image_path)
            except Exception:
                pass
        return jsonify({'error': 'An internal error occurred during generation.', 'detail': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=False)
