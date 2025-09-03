# 🍽️ Food Calorie Prediction Web App

**A web app that predicts the calorie content of food from images, built with Django and TensorFlow/Keras.**

---

## 📌 Overview
This project combines **web development** and **machine learning** to create a system where users can upload food images and receive automatic calorie predictions.  

I experimented with:
- A CNN built from scratch (overfitting issues).  
- Transfer learning with **MobileNetV2**.  
- Optimizations to improve accuracy and generalization.  

---

## 🛠 Tech Stack
- **Backend**: Django  
- **Machine Learning**: TensorFlow/Keras  
- **Frontend**: HTML, CSS  
- **Database**: SQLite (for dev)  
- **Other**: Python  

---

## ✨ Features
- Upload food images through the web interface  
- Predict food type & calories using trained ML models  
- Simple, user-friendly design  

---

## 📂 Project Structure
```bash
├── app/                      # Django app
├── models/                   # Trained models & plots
├── media/food_images/        # Image samples
├── templates/food_calorie_predictor/
├── static/food_calorie_predictor/
├── Report.pdf                 # Internship report
├── requirements.txt           # Project dependencies
├── manage.py
├── .gitignore
