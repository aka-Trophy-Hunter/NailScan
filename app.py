import os
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).parent
YOLO_DIR = REPO_ROOT / "yolov5"
WEIGHTS_PATH = REPO_ROOT / "models" / "best.pt"


@st.cache_resource
def setup_yolov5():
    """Clone the yolov5 repo once (contains segment/predict.py we need)."""
    if not YOLO_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", str(YOLO_DIR)],
            check=True,
        )
    return True


def run_segmentation(image_path: str, out_dir: str):
    script = YOLO_DIR / "segment" / "predict.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--weights", str(WEIGHTS_PATH),
            "--source", image_path,
            "--project", out_dir,
            "--name", "run",
            "--exist-ok",
            "--conf-thres", "0.4",
        ],
        cwd=str(YOLO_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])

    result_dir = Path(out_dir) / "run"
    for f in result_dir.glob("*"):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return f
    return None


st.title("Nail Condition Analyzer")
st.caption("Stage 1: nail detection & segmentation. Disease classification is coming soon.")

with st.spinner("Setting up model..."):
    setup_yolov5()

file = st.file_uploader("Upload a photo of your nail(s)", type=["jpg", "jpeg", "png"])

if file is None:
    st.info("Please upload an image to begin.")
else:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.jpg")
        image.save(input_path)

        with st.spinner("Detecting and segmenting nail region..."):
            try:
                result_path = run_segmentation(input_path, tmpdir + "/out")
            except RuntimeError as e:
                st.error("Segmentation failed. Full error below:")
                st.code(str(e))
                result_path = None

        if result_path:
            st.image(str(result_path), caption="Detected nail region", use_column_width=True)
            st.success("Nail region detected successfully.")
        else:
            st.warning("No nail detected in this image. Try a clearer, closer photo.")

    st.caption(
        "Disease classification (onychomycosis, dystrophy, onycholysis, melanonychia, etc.) "
        "will be added once the classifier model is available."
    )
