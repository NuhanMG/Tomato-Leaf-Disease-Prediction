# Tomato-Leaf-Disease-Prediction

- Model Developed by Nuhan Gunasekara 
- Application Developed by Hathik Ihthizam

## 🍅 ToMate - Deep Learning Tomato Leaf Disease Prediction Model

This project uses deep learning to detect and classify **tomato leaf diseases** from images using a Convolutional Neural Network (CNN). The trained model can later be integrated into the **ToMate - Application** for real-time disease diagnosis.

## 🚀 Features

- Automatically fetches image data from Kaggle using the Kaggle API
- CNN-based image classification using TensorFlow/Keras
- Saves the trained model as `tomate_model.keras`
- Final model is ready to be used in the ToMate - Application folder

---

---

## 🧠 How to Train the Model

1. **Open the Script in Google Colab**  
   Google Colab is highly recommended for training due to free GPU support.

2. **Set Up the Kaggle API**

   The dataset is downloaded from Kaggle directly. To use the API:
   - Go to your Kaggle account settings → [Create New API Token](https://www.kaggle.com/settings/account)
   - Upload the downloaded `kaggle.json` file to your Colab session
   - Run the following setup code:
     ```python
     !mkdir -p ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Run the Entire Python Script**

   The training notebook will:
   - Download and extract the dataset
   - Preprocess the images
   - Train the CNN model
   - Save the trained model as `tomate_model.keras`

4. **Move the Trained Model**

   After training:
   - Move `tomate_model.keras` into the `ToMate - Application` folder
   - This model will be used by the application for prediction

---

