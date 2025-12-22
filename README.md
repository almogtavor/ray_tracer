# Ray Tracer

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Acceleration structure
The renderer always uses an adaptive octree for ray intersections. You can still tune its parameters:

```bash
python ray_tracer.py scenes/example.scene results/example.png --octree-depth 8 --octree-leaf-size 4
```
