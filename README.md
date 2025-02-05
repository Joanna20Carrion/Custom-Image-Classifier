# **Image Classifier Project**

## **Project Overview**  
This project is a deep learning-based **image classifier** that can recognize different categories of images using a **pre-trained convolutional neural network (CNN)**. The model is trained using **PyTorch** and utilizes transfer learning to improve accuracy while reducing training time.  

The dataset used consists of images labeled into multiple classes, and the model is fine-tuned to classify them with high precision.  

## **Project Files**  
The repository contains the following key files:  

- 📁 **`image_classifier_project.ipynb`** – Jupyter Notebook with the training pipeline.  
- 📁 **`train.py`** – Python script for training the model.  
- 📁 **`predict.py`** – Script for making predictions with the trained model.  
- 📁 **`utils.py`** – Utility functions for data preprocessing and visualization.  
- 📁 **`checkpoint.pth`** – The saved trained model.  

## **Installation & Setup**  
To run this project, you need Python 3, PyTorch, and other dependencies. Install them using:  

```bash
pip install -r requirements.txt
```

## **Training the Model**  
To train the model, run:  

```bash
python train.py --data_dir path/to/data --epochs 10 --learning_rate 0.001
```

You can modify **hyperparameters** such as learning rate, batch size, and epochs in `train.py`.  

## **Making Predictions**  
To classify an image using the trained model, use:  

```bash
python predict.py path/to/image checkpoint.pth --top_k 5
```

This will return the **top 5 predicted classes** for the image.  

## **Technologies Used**  
- Python 3  
- PyTorch  
- NumPy  
- Matplotlib  
- Pandas  

## **Contributing**  
... 
