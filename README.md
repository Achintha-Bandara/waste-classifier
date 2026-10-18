# Electronic Component Classifier

An image classification project for identifying electronic components using machine learning and computer vision.

### Download the datasets

The datasets used in this project are hosted on Kaggle and are not included in this repository due to their size.

Visit the Kaggle dataset links below.

| Dataset 1 | [Kaggle dataset link](https://www.kaggle.com/datasets/divyejoshi/pictures-of-simple-electronic-components) |  
| Dataset 2 | [Kaggle dataset link](https://www.kaggle.com/datasets/aryaminus/electronic-components) |  
| Dataset 3 | [Kaggle dataset link](https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification) |  
| Dataset 4 | [Kaggle dataset link](https://www.kaggle.com/datasets/roustoumabdelmoula/textures-dataset) |  
| Dataset 5 | [Kaggle dataset link](https://www.kaggle.com/datasets/ayuraj/asl-dataset) |  
| Dataset 6 | [Kaggle dataset link](https://archive.ics.uci.edu/dataset/908/realwaste) |  

### Training results

Run `modelA.py` from any working directory after placing the RealWaste dataset in
`dataset6/`. The script saves all generated artifacts in `results/modelA/`:

- `model.keras`: trained Keras model
- `metrics.json`: final test loss, accuracy, macro/weighted F1 scores, per-class
  precision/recall/F1, plus final and best validation metrics
- `training_history.json`: loss and accuracy values for every epoch
- `run_metadata.json`: dataset, split, model-run, and class metadata
- `loss_curve.png` and `accuracy_curve.png`: training and validation curves
- `class_distribution.png`: number of images per class

### Confusion matrix

After training, run `plot_confusion_matrix.py` to load `results/modelA/model.keras`
and generate the test-set confusion matrix. The script saves:

- `results/modelA/confusion_matrix.png`: labeled confusion-matrix plot
- `results/modelA/confusion_matrix.csv`: raw confusion-matrix values