# CUB Localization
We adapt an [existing repository](https://github.com/CKCZZJ/Image-Object-Localization)
to evaluate parity models on the Caltech-USCD Birds dataset.

## Quickstart
Simply run:
```bash
python3 coded.py
```

This will print the IoU achieved accross different epochs. Trained models will
be saved to `coded-k2`.

The base model used for generating `src/data/base_bounding-boxes.txt` can be
trained by running:
```bash
python3 train_base_model.py
```
