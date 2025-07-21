# Celebal_Assignment7
# 🌸 Iris Flower Prediction Web App using Streamlit

This project is a simple and interactive **web application** built using **Streamlit**. It predicts the species of an Iris flower using a trained **Random Forest Classifier** and provides users with visual insights into the model and data.

---

## 📊 Dataset

- **Name**: Iris Dataset  
- **Source**: Built-in dataset from `scikit-learn`
- **Features**:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target**:
  - Setosa
  - Versicolor
  - Virginica

---

## 🧠 Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Library**: `scikit-learn`
- The model is trained in `Google Colab` and saved as a `.pkl` file using `joblib`.

---

## 📁 Files

| File Name        | Description                                   |
|------------------|-----------------------------------------------|
| `train_model.py` | Trains and saves the ML model (`iris_model.pkl`) |
| `iris_app.py`    | Streamlit web app using the trained model     |
| `iris_model.pkl` | Saved trained model (downloaded from Colab)   |
| `README.md`      | This documentation file                       |

---

## 🚀 How to Run Locally

1. **Install dependencies**:

   ```bash
   pip install streamlit scikit-learn pandas matplotlib seaborn joblib
