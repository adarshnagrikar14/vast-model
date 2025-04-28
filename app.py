import os
import tempfile
from flask import Flask, request, send_file, jsonify
from predict import Predictor
from pathlib import Path

app = Flask(__name__)

# Initialize the predictor
predictor = Predictor()
predictor.setup()


@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "endpoints": {
            "/generate": "POST - Generate image using form data",
            "/api/generate": "POST - Generate image using JSON data"
        },
        "parameters": {
            "prompt": "Input prompt (used if image analysis is not enabled)",
            "spatial_img": "Reference image file (form endpoint) or image_url (API endpoint)",
            "base_prompt": "Text to prepend to the Gemini-generated prompt",
            "height": "Height of output image (default: 768)",
            "width": "Width of output image (default: 512)",
            "seed": "Random seed (default: 16)",
            "control_type": "LoRA control type (Manhwa or None, default: Manhwa)",
            "lora_scale": "Scale for the LoRA weights (0.0-2.0, default: 0.85)"
        },
        "example_curl": 'curl -X POST http://localhost:8000/api/generate -H "Content-Type: application/json" -d \'{"prompt": "A heroic character", "base_prompt": "Create a detailed illustration with clean lines and vibrant colors", "image_url": "https://example.com/image.jpg", "lora_scale": 0.8}\' --output result.png'
    })


@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get form data
        prompt = request.form.get('prompt', '')
        height = int(request.form.get('height', 768))
        width = int(request.form.get('width', 512))
        seed = int(request.form.get('seed', 16))
        control_type = request.form.get('control_type', 'Manhwa')
        lora_scale = float(request.form.get('lora_scale', 1.25))
        base_prompt = request.form.get('base_prompt', '')

        # Handle file upload to temp file
        spatial_img = None
        if 'spatial_img' in request.files:
            file = request.files['spatial_img']
            if file.filename:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.png')
                file.save(temp_file.name)
                spatial_img = Path(temp_file.name)

        # Generate image
        output_path = predictor.predict(
            prompt=prompt,
            spatial_img=spatial_img,
            base_prompt=base_prompt if base_prompt else None,
            height=height,
            width=width,
            seed=seed,
            control_type=control_type,
            lora_scale=lora_scale
        )

        # Create a temporary file for response
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        with open(output_path, 'rb') as f_in:
            with open(temp_output.name, 'wb') as f_out:
                f_out.write(f_in.read())

        response = send_file(temp_output.name, mimetype='image/png')

        # Clean up the input temp file if it exists
        if spatial_img:
            os.unlink(spatial_img)

        @response.call_on_close
        def cleanup():
            # Clean up both input and output temp files
            if os.path.exists(temp_output.name):
                os.unlink(temp_output.name)

        return response

    except Exception as e:
        # Clean up temp files in case of error
        if 'spatial_img' in locals() and spatial_img:
            if os.path.exists(spatial_img):
                os.unlink(spatial_img)
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        # Get JSON data
        data = request.get_json()
        prompt = data.get('prompt', '')
        height = int(data.get('height', 768))
        width = int(data.get('width', 512))
        seed = int(data.get('seed', 16))
        control_type = data.get('control_type', 'Manhwa')
        lora_scale = float(data.get('lora_scale', 0.85))
        # Always pass base_prompt even if empty
        base_prompt = data.get('base_prompt', '')

        # Handle image URL if provided
        spatial_img = data.get('image_url', None)

        # Generate image
        output_path = predictor.predict(
            prompt=prompt,
            spatial_img=spatial_img,
            base_prompt=base_prompt,
            height=height,
            width=width,
            seed=seed,
            control_type=control_type,
            lora_scale=lora_scale
        )

        # Create a temporary file for response
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        with open(output_path, 'rb') as f_in:
            with open(temp_output.name, 'wb') as f_out:
                f_out.write(f_in.read())

        response = send_file(temp_output.name, mimetype='image/png')

        @response.call_on_close
        def cleanup():
            # Clean up output temp file
            if os.path.exists(temp_output.name):
                os.unlink(temp_output.name)

        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
