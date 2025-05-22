# Before running, install requirements: pip install flask numpy pillow opencv-python

from PIL import Image
import numpy as np
import os
import contextlib
import cv2
import io
import base64
import sys
import json

# chdir to current file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Flask not installed. Please install it using: pip install flask")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "fairseq")))


class ignore_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Create Flask app
app = Flask(__name__)

# Initialize the model
model_initialized = False
visual_grounding = None
load_model = None


def initialize_model():
    global model_initialized, visual_grounding, load_model
    if not model_initialized:
        try:
            # Import here to avoid loading until needed
            from demo import visual_grounding as vg, load_model as lm

            visual_grounding = vg
            load_model = lm

            print("Loading model...", flush=True)
            load_model("./checkpoints/checkpoint_rgbd.pt", "./bert/vocab.txt")
            model_initialized = True
            print("Model loaded successfully", flush=True)
        except ImportError as e:
            print(f"Error importing demo module: {e}")
            sys.exit(1)


def process_image(rgb_base64, depth_base64, message, debug_info=False):
    # Initialize model if not already done
    initialize_model()

    # Decode RGB image
    rgb_data = base64.b64decode(rgb_base64)
    rgb_image = Image.open(io.BytesIO(rgb_data)).convert("RGB")

    # Decode depth image if provided
    depth_image = None
    if depth_base64:
        depth_data = base64.b64decode(depth_base64)
        # Create a temporary BytesIO object
        depth_buffer = io.BytesIO(depth_data)
        # Read the image as numpy array
        depth_file = np.load(depth_buffer, allow_pickle=True)
        depth_image = depth_file

    # Process the image and expression
    with ignore_print() if not debug_info else contextlib.suppress():
        output = visual_grounding(rgb_image, message, depth_image)

    # Convert output mask to base64
    output_mask = Image.fromarray(output[1], mode="L")
    buffer = io.BytesIO()
    output_mask.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return mask_base64


@app.route("/process", methods=["POST"])
def process_request():
    try:
        # Get data from request
        data = request.json
        rgb_base64 = data.get("rgb_image")
        depth_base64 = data.get("depth_image")
        message = data.get("message")
        debug_info = data.get("debug_info", False)

        print(f"Received request with message: {message}", flush=True)

        # Process images and get mask
        mask_base64 = process_image(rgb_base64, depth_base64, message, debug_info)

        # Return the mask
        return jsonify({"mask": mask_base64})
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing request: {error_msg}", flush=True)
        return jsonify({"error": error_msg}), 500


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RES Model Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")

    args = parser.parse_args()

    print(f"Starting server on {args.host}:{args.port}")
    initialize_model()
    app.run(host=args.host, port=args.port, debug=False)
    