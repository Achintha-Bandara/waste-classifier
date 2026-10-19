# Waste Classifier

Image-classification models for the [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) dataset. The project contains two custom convolutional neural networks and two pretrained lightweight-model baselines.

## Dataset

Download the RealWaste dataset and place its class directories directly under `dataset6/`:

```text
dataset6/
  Cardboard/
  Food Organics/
  Glass/
  Metal/
  Miscellaneous Trash/
  Paper/
  Plastic/
  Textile Trash/
  Vegetation/
```

The training scripts discover the class names from these subdirectories. Images are loaded from nested directories when present, resized to `64 x 64`, and split into stratified training, validation, and test sets using a 70/15/15 split. The default random seed is `42`.

## Installation

Use Python with the versions listed in [requirements.txt](requirements.txt):

```powershell
python -m pip install -r requirements.txt
```

The scripts use TensorFlow/Keras, NumPy, Matplotlib, and scikit-learn. The pretrained-model scripts download ImageNet weights on their first run when `--weights imagenet` is used.

## Training the custom models

Each command trains one optimizer configuration for 50 epochs by default. At least 20 epochs are required by the assignment.

### Model A

Model A is the standard convolutional CNN. The default architecture is `deep`; `base` is also available.

```powershell
python modelA.py --optimizer adam
python modelA.py --optimizer adam_1e-3
python modelA.py --optimizer sgd
python modelA.py --optimizer sgd_m0.5
python modelA.py --optimizer sgd_m0.9
```

Additional options include:

```text
--arch base|deep
--aug none|geometric|full
--epochs N
--batch-size N
--seed N
--no-cache
```

### Model B

Model B uses depthwise-separable convolutions and is constrained to fewer than 100,000 trainable parameters.

```powershell
python modelB.py --optimizer adam
python modelB.py --optimizer adam_1e-3
python modelB.py --optimizer sgd
python modelB.py --optimizer sgd_m0.5
python modelB.py --optimizer sgd_m0.9
```

Model B accepts the same `--aug`, `--epochs`, `--batch-size`, `--seed`, and `--no-cache` options as Model A. Both custom-model scripts use `dataset6/` automatically; `--dataset` is retained as an implementation-level override but is not used for the reported experiments.

## Pretrained lightweight models

The pretrained baselines use the same `dataset6/` split and fine-tune ImageNet-pretrained MobileNetV2 and EfficientNet-B0. Each script first trains the classifier head and then fine-tunes the selected top backbone layers.

```powershell
python mobilenet.py
python efficientnet.py
```

Useful options include `--head-epochs`, `--fine-epochs`, `--fine-tune-layers`, `--img-size`, `--head-lr`, `--fine-lr`, `--dropout`, `--weights`, `--aug`, `--batch-size`, and `--seed`. Use `--weights none` to train without ImageNet initialization.

## Results

Each training run writes its outputs below `results/`:

```text
results/
  modelA/<optimizer>/
  modelB/<optimizer>/
  mobilenetv2/
  efficientnetb0/
```

### Approximate recorded performance

The following are approximate test-set results from the saved comparison runs.
Macro precision, macro recall, and macro F1 give each class equal weight, which
is useful here because the RealWaste classes are not perfectly balanced. The
custom-model entries show the strongest recorded optimizer configuration for
each architecture. Exact values can vary with the random seed, hardware,
TensorFlow version, and training settings.

| Model | Configuration | Accuracy | Macro precision | Macro recall | Macro F1 |  
|---|---|---:|---:|---:|---:|  
| Model A | SGD, learning rate `0.01` | ~76.3% | ~0.77 | ~0.76 | ~0.76 |  
| Model B | Adam, learning rate `1e-3` | ~76.0% | ~0.77 | ~0.77 | ~0.77 |  
| MobileNetV2 | ImageNet pretrained | ~79.0% | ~0.80 | ~0.80 | ~0.79 |  
| EfficientNet-B0 | ImageNet pretrained | ~79.0% | ~0.80 | ~0.79 | ~0.79 |  

Model B achieves nearly the same overall accuracy as Model A while using only
about 88,000 parameters and approximately 6.48 million MACs. Its median epoch
time is also substantially lower. MobileNetV2 and EfficientNet-B0 provide the
strongest approximate macro-F1 scores in the saved comparison, but they require
much larger model footprints: approximately 24.1 MB and 40.9 MB, respectively,
compared with approximately 0.78 MB for Model B.

Depending on the model, result directory contains the saved Keras model, metrics, training history, per-class classification report, run metadata, command line, architecture summary, training curves, class-distribution plot, and confusion-matrix files.

## Comparing saved runs

After running the desired experiments, compare the custom models:

```powershell
python compareAB.py
```

This reads saved optimizer runs from `results/modelA/` and `results/modelB/` and writes:

```text
results/comparison_table_AB.md
results/comparison_table_AB.csv
results/comparison_table_AB.json
```

To compare Model B with the pretrained baselines:

```powershell
python compareB.py
```

This writes `comparison_modelB_sota.md`, `comparison_modelB_sota.csv`, and `comparison_modelB_sota.json` under `results/`.

## Reproducibility

The scripts record the command line, arguments, random seed, class mapping, data split information, and training settings in each result directory. For comparable runs, keep the default seed, input resolution, batch size, augmentation mode, and dataset6 split unchanged.
