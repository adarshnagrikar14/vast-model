import numpy as np
from PIL import Image
from predict import Predictor


def create_dummy_image(width, height):
    # Create a random RGB image
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, 'RGB')
    return img


def main():
    # Create dummy subject and mask images
    width, height = 512, 768
    subject_img = create_dummy_image(width, height)
    mask_img = create_dummy_image(width, height)

    # Convert images to bytes
    import io
    subject_bytes = io.BytesIO()
    subject_img.save(subject_bytes, format='PNG')
    subject_bytes = subject_bytes.getvalue()

    mask_bytes = io.BytesIO()
    mask_img.save(mask_bytes, format='PNG')
    mask_bytes = mask_bytes.getvalue()

    # Load model and generate image
    predictor = Predictor()
    predictor.setup()
    output_path = predictor.predict(
        input_image_bytes=subject_bytes,
        mask_image_bytes=mask_bytes,
        expression="k-pop happy"
    )
    print(f"Generated image saved at: {output_path}")


if __name__ == "__main__":
    main()
