# Data

## Training data
We use the Find3D data engine to construct the training set. See [Find Any Part in 3D (Find3D)](https://github.com/ziqi-ma/Find3D) for details.

Training data can be downloaded from [Hugging Face](https://huggingface.co/datasets/patchalign3d/patchalign3d-training-data).

The `core/` folder contains the point clouds, labels, and precomputed DINOv2 patch features.
The optional `renderings/` folder contains the rendered views used to generate labels and to compute 2D visual encoder patch features.

## Evaluation data
- **ShapeNetPart:** [download](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)
- **FAUST (coarse parts from SATR):** [SATR](https://github.com/Samir55/SATR)
- **PartNet-E:** [PartSLIP](https://colin97.github.io/PartSLIP_page/)
- **ScanObjectNN:** [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)
