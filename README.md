# DiabetesPredictor


# 🧪 Diabetes Risk Predictor (3-Feature Logistic Regression)

This is a simple **Streamlit web app** that predicts a person's risk of having diabetes based on three key health features: **Glucose Level**, **BMI**, and **Age**. The model is implemented **from scratch using NumPy**, and is trained on the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

<div align="center">
    <a href="https://www.linkedin.com/in/abdullah-owais-23750920b" target="_blank">
        <img src="https://img.shields.io/badge/-LinkedIn-blue?logo=linkedin&style=flat-square">
    </a>
    <a href="https://github.com/abd-04" target="_blank">
        <img src="https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square">
    </a>
</div>

---

## 🚀 Features

* ⚙️ **Supervised Learning Classification** model trained on labelled data to predict categorical outputs.
* 🔢 **Logistic Regression** algorithm applied from scratch (no sklearn or TensorFlow)
* ⚙️ Uses only 3 interpretable inputs: `Glucose`, `BMI`, and `Age`
* 📈 **Training loss curve** visualization
* 🧠 **Sigmoid function** visualization
* ✅ Live prediction with probability-based interpretation
  

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/abd-04/DiabetesPredictor
cd DiabetesPredictor
pip install -r requirements.txt
```

---

## 📁 Files in the Repo

| File                 | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `app.py`             | Streamlit web app                                              |
| `diabetes.csv`       | Dataset used for training                                      |
| `diabetesanalysis.py`| Script to train the logistic regression model from scratch     |
| `logistic_model.pkl` | Saved model containing weights, bias, scaler, and loss history |
| `README.md`          | You're reading it!                                             |

---

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

Made by **Abdullah Owais** | ML from Scratch

* 🧑‍💼 [LinkedIn](https://www.linkedin.com/in/abdullah-owais-23750920b)
* 🐙 [GitHub](https://github.com/abd-04)

---


