# LungLens

Lung nodule detection from 3D chest CT scans.

## Features

- LUNA16 CT preprocessing
- YOLO-based nodule detection
- 3D candidate merging
- False-positive reduction experiments
- FROC evaluation and error analysis

## Current status

V5 validation results on 11 CT scans:
- Sensitivity: 79.55%
- False positives per scan: 3.455
- Detected nodules: 35 / 44

These are validation results, not independent test or clinical results.

## Dataset

LUNA16. Raw medical images are not included in this repository.

## Requirements

Python 3.12 and a compatible PyTorch/CUDA environment.
