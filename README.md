# Ray Tracer

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Acceleration structures
The renderer now supports multiple spatial acceleration structures that can be selected at runtime:

- `--accel bvh` (default) — classic bounding volume hierarchy.
- `--accel grid` — uniform voxel grid with configurable resolution (`--grid-cells`).
- `--accel octree` — adaptive octree with configurable max depth (`--octree-depth`) and leaf size (`--octree-leaf-size`).

Example:

```bash
python ray_tracer.py scenes/example.scene results/example.png --accel grid --grid-cells 48
```
