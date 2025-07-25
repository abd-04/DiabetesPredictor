# DiabetesPredictor
Here’s a **GitHub-style `README.md`** for your Diabetes Risk Predictor project:

---

# 🧪 Diabetes Risk Predictor (3-Feature Logistic Regression)

This is a simple **Streamlit web app** that predicts a person's risk of having diabetes based on three key health features: **Glucose Level**, **BMI**, and **Age**. The model is implemented **from scratch using NumPy**, and is trained on the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

<div align="left">
    <a href="https://www.linkedin.com/in/abdullah-owais-23750920b" target="_blank">
        <img src="https://img.shields.io/badge/-LinkedIn-blue?logo=linkedin&style=flat-square">
    </a>
    <a href="https://github.com/abd-04" target="_blank">
        <img src="https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square">
    </a>
</div>

---

## 🚀 Features

* 🔢 **Logistic Regression** built from scratch (no sklearn or TensorFlow)
* ⚙️ Uses only 3 interpretable inputs: `Glucose`, `BMI`, and `Age`
* 📈 **Training loss curve** visualization
* 🧠 **Sigmoid function** visualization
* ✅ Live prediction with probability-based interpretation
* 🎨 Built with **Streamlit** for fast web deployment

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/abd-04/diabetes-risk-predictor
cd diabetes-risk-predictor
pip install -r requirements.txt
```

---

## 📁 Files in the Repo

| File                 | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `app.py`             | Streamlit web app                                              |
| `diabetes.csv`       | Dataset used for training                                      |
| `train_model.py`     | Script to train the logistic regression model from scratch     |
| `logistic_model.pkl` | Saved model containing weights, bias, scaler, and loss history |
| `README.md`          | You're reading it!                                             |

---

## 🧠 How it Works

1. The model is trained using:

   * Gradient descent
   * Binary cross-entropy loss
2. Only `Glucose`, `BMI`, and `Age` features are selected
3. Data is standardized using `StandardScaler`
4. A NumPy-based logistic regression model is trained and saved
5. Streamlit app takes user input, applies same scaling, and predicts:

```python
z = np.dot(X_scaled, weights) + bias
prob = 1 / (1 + np.exp(-z))
```

---

## 📊 Sample Visuals

### 🟠 Loss Curve

Plots loss per epoch during training.

### 🧪 Sigmoid Curve

Shows how the model maps raw scores to probabilities using the sigmoid function.

---

## 🔮 Example Prediction Output

* Low Risk: ✅ "You're likely safe"
* Moderate Risk: ⚠️ "Consider regular checkups"
* High Risk: 🚨 "Consult a doctor immediately"

---

## ✨ Demo

You can run the app locally:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

Made with ❤️ by **Abdullah Owais**

* 🧑‍💼 [LinkedIn](https://www.linkedin.com/in/abdullah-owais-23750920b)
* 🐙 [GitHub](https://github.com/abd-04)

---

Let me know if you’d like a **preview image**, a **gif of the app**, or a **badge for deployment (e.g., Streamlit Sharing)** added too!
