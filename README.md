# **Image Classifier Project**

## **Project Overview**  
This project is a deep learning-based **image classifier** that can recognize different categories of images using a **pre-trained convolutional neural network (CNN)**. The model is trained using **PyTorch** and utilizes transfer learning to improve accuracy while reducing training time.  

## **Project Files**  
The repository contains the following key files:  

- 📁 **`image_classifier_project.ipynb`** – Jupyter Notebook with the training pipeline.  
- 📁 **`train.py`** – Python script for training the model.  
- 📁 **`predict.py`** – Script for making predictions with the trained model.  

## **Training the Model**  
To train the model, run:  

```bash
python train.py "flowers" --save_dir "path/to/directory" --arch "vgg13" --lrn 0.001 --hidden_units 2048 --epochs 5 --GPU "GPU"
```

You can modify **hyperparameters** such as learning rate, batch size, and epochs in `train.py`.  

## **Making Predictions**  
To classify an image using the trained model, use:  

```bash
python predict.py "path/to/image.jpg" --load_dir "checkpoint.pth" --top_k 5 --category_names "cat_to_name.json" --GPU "GPU"
```

This will return the **top 5 predicted classes** for the image.  

## **Technologies Used**  
- Python 3  
- PyTorch  
- NumPy  
- Matplotlib  
- Pandas  

## **Author**  
**Joanna Alexandra Carrión Pérez**: Bachelor's degree in Electronic Engineering. Passionate about Data Science and Artificial Intelligence. [LinkedIn](https://www.linkedin.com/in/joanna-carrion-perez/)  

## **Contact**  
For any questions or suggestions, feel free to contact me at **joannacarrion14@gmail.com**.  

## **Contributions**  
Contributions are welcome! If you have ideas or improvements, feel free to fork the repository and submit a pull request.  
