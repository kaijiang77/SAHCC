# Data

Large datasets are not stored in Git.

Place ShanghaiTech Part A here before preprocessing:

```text
data/ShanghaiTech/part_A/
  train_data/
    images/
    ground-truth/
  test_data/
    images/
    ground-truth/
```

Build the project dataset layout with:

```bash
python tools/prepare_shha.py --overwrite
```

The generated `data/unified/SHHA/` directory is local runtime data and is ignored by Git.
