# Plant Disease Detection

Train a convolutional neural network that spots tomato leaf diseases
from photos.

## What it does

Given a photo of a tomato leaf, the model classifies it as healthy,
bacterial spot or early blight.

## Data

PlantVillage (CC0, Hughes and Salathe 2015), the standard public dataset
for this task. The full dataset is about 800 MB, so this project uses a
small subset: 90 images each of 3 tomato classes (270 images total),
downloaded automatically by `download_data.py` from the official public
GitHub repo (spMohanty/PlantVillage-Dataset). No login needed. Images
are 256x256 and resized to 128x128 for training.

## Model

- A small CNN: 3 conv blocks (32, 64, 128 filters), a 128 unit dense
  layer, dropout 0.3.
- Data augmentation: horizontal flips and small rotations.
- Trained for 12 epochs on 128x128 images.
- The train/validation split is stratified, so every class appears in
  both sets (216 train, 54 validation, 18 per class in validation).

## Results

- **Validation accuracy: 79.6 percent** (43 of 54 validation images correct).
- Per class F1: healthy 0.889, bacterial spot 0.750, early blight 0.750.
- The plot `training_history.png` shows training and validation accuracy
  climbing together across the 12 epochs.

This is a deliberately small setup: 270 images and 12 epochs on a CPU.
With the full PlantVillage dataset and more epochs the same architecture
scores much higher, and the code is structured so swapping in more data
is just a matter of changing the numbers in `download_data.py`.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

`main.py` downloads the data itself on the first run (or run
`python download_data.py` separately).
