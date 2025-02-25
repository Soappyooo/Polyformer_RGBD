from PIL import Image
import numpy as np
import os
import contextlib
import cv2

# chdir to current file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "fairseq")))


class ignore_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main(input_img_file, msg_file, depth_img_file=None, output_image_path="./output", debug_info=False):
    os.makedirs(output_image_path, exist_ok=True)

    from demo import visual_grounding, load_model
    load_model("./checkpoints/checkpoint_rgbd.pt", "./bert/vocab.txt")
    
    while True:
        print("Press enter to continue")
        input()
        image = Image.open(input_img_file).convert("RGB")
        print(f"using image: {os.path.abspath(input_img_file)}", flush=True)
        if depth_img_file is not None:
            depth = cv2.imread(depth_img_file, cv2.IMREAD_UNCHANGED)
            depth.dtype = np.float16
            print(f"using depth image: {os.path.abspath(depth_img_file)}", flush=True)
        with open(msg_file, "r") as f:
            msg = f.read()
            print(f"using message: {os.path.abspath(msg_file)}", flush=True)
        print(f"received message: {msg}", flush=True)
        with ignore_print() if not debug_info else contextlib.suppress():
            output = visual_grounding(image, msg, depth if depth_img_file is not None else None)
        output_image = Image.fromarray(output[0])
        output_mask = Image.fromarray(output[1], mode="L")
        output_image.save(os.path.join(output_image_path, "output_rgb.jpg"))
        output_mask.save(os.path.join(output_image_path, "output_mask.png"))
        print("output saved", flush=True)


if __name__ == "__main__":
    main("./demo/rgb_demo.jpg", "./demo/msg_demo.txt", "./demo/depth_demo.png", "./output", debug_info=True)
