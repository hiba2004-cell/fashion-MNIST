# 👗 Fashion MNIST Classifier – CNN Project

## 🧠 Description du projet
Ce projet implémente un **modèle DEEPLEARNING** pour la **classification d’images du dataset Fashion MNIST**.  
L’objectif est de reconnaître automatiquement le type de vêtement (ex. : t-shirt, pantalon, pull, basket, robe, etc.) à partir d’images en niveaux de gris de 28x28 pixels.

Le projet est réalisé avec **TensorFlow/Keras**.

---

## 📂 Structure du projet

```
Fashion-MNIST-Classifier/
│
├── dataset/                # Dataset téléchargé automatiquement via Keras
│
├── models/
│   └── fashion_cnn_model.h5
│
├── notebooks/
│   └── fashion_mnist_classifier.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── app.py
```

---

## 🧩 Architecture du modèle CNN

Le modèle utilise :  
 
- **1 couche Flatten**  
- **3 couches Dense** dont une Softmax  


```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## ⚙️ Environnement & dépendances

### Installation
```bash
git clone https://github.com/username/Fashion-MNIST-Classifier.git
cd Fashion-MNIST-Classifier
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow==2.15.0
numpy
matplotlib
pandas
scikit-learn
streamlit
```

---

## 📊 Résultats obtenus

| Métrique | Valeur |
|-----------|---------|
| **Accuracy** | 0.92 |
| **Loss** | 0.18 |

---


## 📈 Améliorations possibles

- Utilisation du **Transfer Learning** (MobileNet, EfficientNet)  
- Amélioration du preprocessing (normalisation, augmentation)  
- Optimisation GPU et quantization pour mobile

---

## 👩‍💻 Auteur
**Nom :** Hiba Nadiri  
**École :** ENSA El Jadida  
**Projet :** Classification des vêtements avec CNN (Fashion MNIST)  
**Date :** Octobre 2025

---

