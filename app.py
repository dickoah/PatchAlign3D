import gradio as gr
import os
import shutil
import logging
from huggingface_hub import snapshot_download
import sys
import traceback
import plotly.graph_objects as go
import numpy as np

# Add 'src' to path to allow importing 'tools.patch_align_segmenter'
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from src.tools.patch_align_segmenter import PatchAlignSegmenter  # type: ignore
except ImportError:
    try:
        from tools.patch_align_segmenter import PatchAlignSegmenter  # type: ignore
    except ImportError as e:
        print(f"Failed to import PatchAlignSegmenter: {e}")
        traceback.print_exc()
        PatchAlignSegmenter = None

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("patchalign_app")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)

# Initialize Segmenter Global
segmenter = None
init_error = None

if PatchAlignSegmenter:
    logger.info("Initializing PatchAlignSegmenter...")
    try:
        segmenter = PatchAlignSegmenter()
        logger.info("Segmenter initialized successfully.")
    except Exception as e:
        init_error = str(e)
        logger.error(f"Failed to initialize segmenter: {e}")
        traceback.print_exc()
else:
    init_error = "PatchAlignSegmenter class could not be imported."
    logger.error("PatchAlignSegmenter class not available.")


def _build_pointcloud_plot(pc_data):
    """Build a plotly 3D scatter from point cloud data."""
    pts = pc_data["points"]
    labels = pc_data["labels"]
    label_names = pc_data["label_names"]
    colors = pc_data["colors"]

    fig = go.Figure()
    for i, name in enumerate(label_names):
        mask = labels == i
        count = mask.sum()
        if count == 0:
            continue
        rgb = (colors[mask][0] * 255).astype(int)
        color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        fig.add_trace(go.Scatter3d(
            x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
            mode="markers",
            marker=dict(size=3, color=color_str, opacity=0.9),
            name=f"{name} ({count})",
            hovertemplate=f"<b>{name}</b><br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgb(20,20,20)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing="constant", font=dict(size=14)),
        paper_bgcolor="rgb(30,30,30)",
        font=dict(color="white"),
        height=650,
    )
    return fig


def segment_mesh(mesh_path: str, expected_parts_str: str, density: int):
    """
    Run PatchAlign3D segmentation on a mesh and return the labeled point cloud.
    """
    if init_error:
        raise gr.Error(f"Initialization Error: {init_error}")
        
    logger.info(f"Processing mesh: {mesh_path}")
    logger.info(f"Expected parts: {expected_parts_str}, density: {density}")
    
    if not mesh_path:
        return None
    
    if not segmenter:
        raise gr.Error("Segmenter failed to initialize. Check server logs.")

    if not expected_parts_str or not expected_parts_str.strip():
        raise gr.Error("Please enter at least one part label.")

    prompts = [p.strip() for p in expected_parts_str.split(",") if p.strip()]
    if not prompts:
        raise gr.Error("Please enter at least one part label.")

    try:
        result = segmenter.segment(mesh_path, prompts, display_points=int(density))
        pc_fig = _build_pointcloud_plot(result["point_cloud"])
        return pc_fig
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        traceback.print_exc()
        raise gr.Error(f"Segmentation failed: {str(e)}")


# Build the UI
with gr.Blocks(title="PatchAlign3D - Segmentation") as demo:
    gr.Markdown("## PatchAlign3D Segmentation Demo")
    
    with gr.Row():
        # Left Panel: Input
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            
            input_model = gr.Model3D(label="Input Mesh (.glb)")
            
            expected_parts = gr.Textbox(
                label="Expected Parts", 
                placeholder="e.g., seat, back, leg, arm",
                info="Comma-separated part names for the model to find."
            )
            
            density_slider = gr.Slider(
                minimum=2048, maximum=50000, value=20000, step=1000,
                label="Point Cloud Density",
                info="Number of display points (model uses 2048; extras are NN-propagated)."
            )
            
            generate_btn = gr.Button("Run Segmentation", variant="primary")
            
        # Right Panel: Point Cloud Result
        with gr.Column(scale=2):
            gr.Markdown("### Labeled Point Cloud")
            pc_plot = gr.Plot(label="Segmentation Result")

    # Wire events
    generate_btn.click(
        fn=segment_mesh,
        inputs=[input_model, expected_parts, density_slider],
        outputs=[pc_plot]
    )

if __name__ == "__main__":
    demo.launch()
