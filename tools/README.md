# Tools

Only `prepare_shha.py` is part of the public project.

It converts ShanghaiTech Part A into the unified SHHA layout used by the training and evaluation configs:

```bash
python tools/prepare_shha.py \
  --source-dir /path/to/ShanghaiTech/part_A \
  --output-root data
```

Other local dataset-conversion and scratch scripts are intentionally ignored.
