import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import open_clip
import matplotlib.pyplot as plt
from easydict import EasyDict
from typing import List, Optional, Dict, Any
from scipy.spatial import cKDTree
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PatchAlignSegmenter")

# Add 'src' to path to import models
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# CURRENT_DIR is .../src/tools
# We want to add .../src to sys.path to allow 'from models import ...'
SRC_ROOT = os.path.dirname(CURRENT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

try:
    from models import point_transformer
except ImportError:
    # If that fails, maybe we are running from root and 'src' is a package
    try:
        from src.models import point_transformer
    except ImportError:
        logger.error("Could not import 'models.point_transformer'. Check PYTHONPATH.")
        raise

# Configuration constants
MODEL_REPO_ID = "patchalign3d/patchalign3d-encoder"
MODEL_FILENAME = "patchalign3d.pt"
CLIP_MODEL_NAME = "ViT-bigG-14"
CLIP_PRETRAINED = "laion2b_s39b_b160k"

PART_ONLY_TEMPLATES = ["{}", "a {}", "{} part"]


def _clean_text(s):
    s = s.strip().lower().replace("_", " ")
    return " ".join(s.split())


class PatchToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb):
        x = patch_emb.transpose(1, 2)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class PatchAlignSegmenter:
    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize the PatchAlignSegmenter.
        Args:
            checkpoint_path (str, optional): Path to the model checkpoint.
                                             If None, downloads from Hugging Face.
            device (str, optional): Device to run on ('cuda' or 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Path setup
        if checkpoint_path is None:
            # Navigate up from src/tools to project root, then to data/model
            # CURRENT_DIR is .../src/tools
            project_root = os.path.dirname(os.path.dirname(CURRENT_DIR))
            data_dir = os.path.join(project_root, "data", "model")
            
            os.makedirs(data_dir, exist_ok=True)
            ckpt_file = os.path.join(data_dir, MODEL_FILENAME)
            
            if not os.path.exists(ckpt_file):
                logger.info(f"Downloading model checkpoint to {data_dir}...")
                snapshot_download(repo_id=MODEL_REPO_ID, local_dir=data_dir)
            self.checkpoint_path = ckpt_file
        else:
            self.checkpoint_path = checkpoint_path

        # Load Models
        self._load_models()

    def _load_models(self):
        logger.info("Loading CLIP model...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        
        # Get text projection dimension safely
        if hasattr(self.clip_model, "text_projection"):
            self.text_dim = self.clip_model.text_projection.shape[1]
        else:
            self.text_dim = 1280  # Default for ViT-bigG-14 if attribute missing

        logger.info("Loading PointTransformer and Projection Head...")
        # Configuration matching infer.py defaults
        self.cfg = EasyDict(
            trans_dim=384,
            depth=12,
            drop_path_rate=0.1,
            cls_dim=50,
            num_heads=6,
            group_size=32,
            num_group=128,
            encoder_dims=256,
            color=False,
            num_classes=16,
        )
        
        self.point_model = point_transformer.get_model(self.cfg).to(self.device)
        self.proj_head = PatchToTextProj(in_dim=384, out_dim=self.text_dim).to(self.device)

        # Load checkpoint weights
        logger.info(f"Loading weights from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if "model" in ckpt:
            self.point_model.load_state_dict(ckpt["model"], strict=False)
        if "proj" in ckpt:
            self.proj_head.load_state_dict(ckpt["proj"], strict=False)
        
        self.point_model.eval()
        self.proj_head.eval()

    def _encode_text_prompts(self, prompt_list: List[str]):
        """Encode text prompts using CLIP."""
        texts = []
        for prompt in prompt_list:
            clean_prompt = _clean_text(prompt)
            for tpl in PART_ONLY_TEMPLATES:
                texts.append(tpl.format(clean_prompt))
        
        with torch.no_grad():
            toks = self.tokenizer(texts).to(self.device)
            feat = self.clip_model.encode_text(toks)
            feat = F.normalize(feat, dim=-1)
        
        # Aggregate templates per label (mean pooling)
        per_label_feats = []
        idx = 0
        n_templates = len(PART_ONLY_TEMPLATES)
        for _ in prompt_list:
            chunk = feat[idx : idx + n_templates]
            per_label_feats.append(F.normalize(chunk.mean(dim=0, keepdim=True), dim=-1))
            idx += n_templates
            
        return torch.cat(per_label_feats, dim=0)

    def _normalize_points(self, points):
        """Prepare points: ensure (N,3) and normalize to unit sphere."""
        # Simple centering and scaling
        centroid = np.mean(points, axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale
        return points

    def segment(self, mesh_path: str, prompt_list: List[str], output_path: Optional[str] = None, display_points: int = 20000):
        """
        Segment a mesh using the provided text prompts.
        
        Args:
            mesh_path (str): Path to input mesh (.glb, .obj, .ply).
            prompt_list (List[str]): List of parts to segment (e.g. ["chair back", "seat legs", ...]).
            output_path (str): Path to save the result. If None, saves next to input.
            
        Returns:
            dict with keys:
                'mesh_path': str - Path to the segmented mesh file.
                'point_cloud': dict - Point cloud data for visualization:
                    'points': np.ndarray (N, 3) - sampled point positions
                    'labels': np.ndarray (N,) - integer label per point
                    'label_names': List[str] - prompt name per label index
                    'colors': np.ndarray (N, 3) - RGB float colors per point
        """
        logger.info(f"Segmenting {mesh_path} with prompts: {prompt_list}")
        
        # 1. Load Mesh
        mesh = trimesh.load(mesh_path, force="mesh")
        
        # 2. Sample Points
        N_POINTS = 2048
        sampled_points, _ = trimesh.sample.sample_surface(mesh, N_POINTS)
        
        # 3. Preprocess for Inference
        # Based on infer.py: Y/Z swap might be implicitly handled by data or training coords.
        # Here we just center/normalize.
        input_points = self._normalize_points(sampled_points).astype(np.float32)
        
        # Prepare for model: (B, 3, N) - transpose coordinates
        # infer.py does: transpose(2, 1) and swaps Y/Z: pts[:, [1, 2], :] = pts[:, [2, 1], :]
        points_tensor = torch.from_numpy(input_points).unsqueeze(0) # (1, N, 3)
        points_tensor = points_tensor.transpose(2, 1).contiguous()   # (1, 3, N)
        
        # Apply Y/Z swap matching infer.py logic
        points_tensor[:, [1, 2], :] = points_tensor[:, [2, 1], :]
        points_tensor = points_tensor.to(self.device)

        # 4. Inference
        with torch.no_grad():
            # Get Patch Features
            patch_emb, patch_centers, patch_idx = self.point_model.forward_patches(points_tensor)
            patch_feat = self.proj_head(patch_emb)
            
            # Get Text Features
            text_feat = self._encode_text_prompts(prompt_list)
            
            # Compute Similarity
            logits = (patch_feat @ text_feat.t()) / 0.07 # clip_tau default
            
            # Assign labels to patches (simple max)
            patch_labels = logits.argmax(dim=-1) # (B, NumPatches)

            # Map patch labels back to points (Nearest Patch Center assignment)
            # Similar to assign_points_from_patches with 'nearest' mode in infer.py
            # But simpler here: we just iterate and assign.
            
            # Reconstruct point labels
            # Unswap coords for KDTree query to match input points
            # Actually patch_centers are also Y/Z swapped.
            # Best is to just use KNN in the swapped space.
            
            target_points = points_tensor[0].transpose(1, 0) # (N, 3)
            centers = patch_centers[0].transpose(1, 0)       # (K, 3)
            
            # Use PyTorch for quick nearest patch search
            # dist (N, K)
            dist = torch.cdist(target_points, centers)
            nearest_idx = dist.argmin(dim=1) # (N,) index of nearest center
            
            point_labels = patch_labels[0][nearest_idx].cpu().numpy()

        # 5. Build discrete color palette for labels
        n_labels = len(prompt_list)
        cmap = plt.get_cmap("tab10" if n_labels <= 10 else "tab20")
        label_colors_rgb = np.array([cmap(i / max(n_labels - 1, 1))[:3] for i in range(n_labels)])

        # 6. Densify: sample more points on the surface and propagate labels via NN
        #    The model only labels 2048 points; we use those as anchors.
        if display_points > N_POINTS:
            dense_points, _ = trimesh.sample.sample_surface(mesh, display_points)
            kdtree_dense = cKDTree(sampled_points)
            _, nn_idx = kdtree_dense.query(dense_points, k=1)
            dense_labels = point_labels[nn_idx]
            display_pts = dense_points
            display_labels = dense_labels
        else:
            display_pts = sampled_points
            display_labels = point_labels

        # Color the display point cloud
        pc_colors = label_colors_rgb[display_labels]  # (display_points, 3) float

        # 7. Propagate Labels to Mesh Vertices (Simple NN)
        kdtree = cKDTree(sampled_points)
        _, vertex_indices = kdtree.query(mesh.vertices, k=1)
        vertex_labels = point_labels[vertex_indices]
        
        # Colorize Mesh Vertices
        vertex_colors_rgb = label_colors_rgb[vertex_labels]  # (V, 3) float
        vertex_colors_rgba = np.hstack([
            (vertex_colors_rgb * 255).astype(np.uint8),
            np.full((len(vertex_colors_rgb), 1), 255, dtype=np.uint8)
        ])
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors_rgba)

        # 8. Export
        if output_path is None:
            base, ext = os.path.splitext(mesh_path)
            output_path = f"{base}_segmented.glb"
            
        mesh.export(output_path)
        logger.info(f"Saved segmented mesh to {output_path}")
        
        return {
            "mesh_path": output_path,
            "point_cloud": {
                "points": display_pts,
                "labels": display_labels,
                "label_names": prompt_list,
                "colors": pc_colors,
            },
        }
