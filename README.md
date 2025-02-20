# **Custom Image Classifier**

This repository contains a **PyTorch**-based **image classification** project, focused on recognizing different flower categories (or any other image dataset you wish to use). It includes scripts to train a model and to predict the class of a new image, as well as a Jupyter notebook for interactive use.

---

## Repository Contents

- **`assets/`**  
  A folder to store additional resources (sample images, graphics, etc.). It no longer contains the Jupyter notebook that was removed to keep the repository tidy.

- **`cat_to_name.json`**  
  A JSON file that maps class IDs (e.g., `1`, `2`, `3`, ...) to flower names (e.g., `'pink primrose'`, `'hard-leaved pocket orchid'`, etc.). Feel free to modify it if you wish to train the model on different categories.

- **`image_classifier_project.ipynb`**  
  A Jupyter notebook providing a complete workflow for training and prediction. This is useful for:
  - Loading and preprocessing your dataset.
  - Defining and training the model.
  - Evaluating performance.
  - Making predictions in an interactive manner.

- **`predict.py`**  
  A Python script for **predicting** the category of an image using a previously trained model. You can specify:
  - The path to the image you want to classify.
  - The checkpoint file containing the trained model.
  - The number of most likely classes to display.
  - The JSON file for mapping class indices to real names.
  - The computing device (CPU or GPU).

- **`train.py`**  
  A Python script for **training** a model with PyTorch. It allows you to configure:
  - The base architecture (`vgg13` or `alexnet`).
  - The learning rate.
  - The number of hidden units in the final layer.
  - The number of epochs.
  - The computing device (CPU or GPU).
  - The directory where the trained model is saved.

- **`README.md`**  
  This file, which describes the project contents and usage.

---

## Requirements

Since there is no `requirements.txt` file, you must install dependencies manually. It is recommended to create a virtual environment (e.g., with `venv` or `conda`) and install the following packages:

- **Python 3.x**
- **PyTorch** and **Torchvision** (compatible with your Python version and OS)
- **PIL (Pillow)**
- **NumPy**
- **pandas**
- **matplotlib**
- **argparse**

Example installation using `pip`:

```bash
pip install torch torchvision pillow numpy pandas matplotlib
```

---

## Usage

### 1. Training

To train the model, organize your dataset in a directory containing `train`, `valid`, and `test` subfolders (e.g., `flowers/`). Then, run:

```bash
python train.py "flowers" --save_dir "path/to/directory" --arch "vgg13" --lrn 0.001 --hidden_units 2048 --epochs 5 --GPU "GPU"
```

- **`"flowers"`**: Path to your main data directory, which should contain the `train`, `valid`, and `test` folders.
- **`--save_dir "path/to/directory"`**: Directory where the trained model is saved as `checkpoint.pth`.
- **`--arch "vgg13"`**: The architecture to use (`vgg13` or `alexnet`).
- **`--lrn 0.001`**: The learning rate.
- **`--hidden_units 2048`**: Number of hidden units in the final classifier layer.
- **`--epochs 5`**: Number of training epochs.
- **`--GPU "GPU"`**: Use the GPU (if available and PyTorch is configured with CUDA). If you prefer CPU, remove this argument or set it to `"CPU"`.

### 2. Prediction

Once the model is trained, you can predict the class of an image by running:

```bash
python predict.py "path/to/image.jpg" --load_dir "checkpoint.pth" --top_k 5 --category_names "cat_to_name.json" --GPU "GPU"
```

- **`"path/to/image.jpg"`**: Path to the image you want to classify.
- **`--load_dir "checkpoint.pth"`**: The file containing the trained model (generated during training).
- **`--top_k 5`**: The number of most likely predictions to display.
- **`--category_names "cat_to_name.json"`**: A JSON file mapping class indices to real names.
- **`--GPU "GPU"`**: Use the GPU. If you prefer CPU, change this to `"CPU"` or omit it.

### 3. Interactive Use (Optional)

If you prefer a more educational and visual approach:

1. Open the **`image_classifier_project.ipynb`** notebook in Jupyter:
   ```bash
   jupyter notebook image_classifier_project.ipynb
   ```
2. Run the cells in order to:
   - Load your dataset.
   - Define the architecture.
   - Train the model.
   - Make predictions.

---

## Contributions

Contributions are welcome! If you find bugs or would like to improve this project:

1. Fork this repository.
2. Create a new branch for your contribution.
3. Submit a pull request describing your changes.

---

## **Author**  
**Joanna Alexandra Carrión Pérez**: Bachelor's degree in Electronic Engineering. Passionate about Data Science and Artificial Intelligence. [LinkedIn](https://www.linkedin.com/in/joanna-carrion-perez/)  

---

## **Contact**  
For any questions or suggestions, feel free to contact me at **joannacarrion14@gmail.com**.  
