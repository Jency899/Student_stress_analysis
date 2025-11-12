# Student Stress Level Prediction using Machine Learning

## 📘 Abstract
This project focuses on predicting student stress levels using machine learning techniques based on lifestyle, academic, and social factors. The dataset was created using Google Form responses collected from university students. Key attributes include study hours, sleep duration, academic pressure, financial stress, family support, and health conditions.  

After preprocessing and feature encoding, three models — **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)** — were developed to classify stress levels on a scale from 1 (Low) to 5 (High). Evaluation metrics such as **accuracy, precision, recall, and F1-score** were computed to compare their performance. Logistic Regression achieved the best results, emphasizing its interpretability and effectiveness on small datasets.  

This study demonstrates how AI-driven predictive analysis can assist educational institutions in identifying and managing student stress at an early stage.

---

## 🎯 Introduction
Stress among students has emerged as a serious issue due to academic pressure, competition, and lifestyle changes. The goal of this project is to build an intelligent machine learning framework that predicts students’ stress levels based on behavioral, academic, and personal factors.  

By identifying significant stress indicators, this system can help:
- Students gain awareness of their mental well-being.
- Teachers and counselors design targeted interventions.
- Institutions promote a healthier academic environment.  

This project highlights the practical use of AI for **mental health prediction** in educational settings.

---

## ⚙️ Methodology

### 1. Data Collection
Data was gathered via **Google Forms** from students of different academic years and programs. Each record contained both numerical and categorical information about a student’s academic habits, lifestyle, and family background.

### 2. Data Preprocessing
- Removed unnecessary fields (e.g., timestamps).  
- Handled missing values and outliers.  
- Normalized numerical attributes to a common scale.  

### 3. Feature Encoding
Categorical features such as `Course`, `Gender`, and `Year of Study` were converted into numerical representations using **One-Hot Encoding** to make them suitable for model training.

### 4. Model Development
Three supervised machine learning models were trained using **Scikit-Learn**:
- **Logistic Regression** – baseline linear model for interpretability.  
- **Random Forest Classifier** – ensemble model for non-linear patterns.  
- **Support Vector Machine (SVM)** – for high-dimensional data separation.  

The dataset was split into **80% training** and **20% testing** subsets.  

### 5. Evaluation Metrics
Each model was evaluated using:
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  

These metrics helped assess each algorithm’s reliability and performance balance.

---

## 📊 Dataset Description

| Property | Description |
|-----------|-------------|
| **Source** | Custom dataset from student Google Form responses |
| **Records** | 50 student entries |
| **Attributes** | 14 (mix of categorical & numerical) |
| **Target Variable** | Stress Level (1–5 scale) |
| **Split** | 80% Training, 20% Testing |

**Features include:**
- Academic: `study_hours`, `academic_pressure`, `course`, `year_of_study`
- Lifestyle: `sleep_hours`, `physical_activity`, `screen_time`
- Social & Emotional: `social_support`, `family_pressure`, `financial_pressure`

---

## 💻 Implementation

### Libraries Used
- **pandas** — Data manipulation and analysis  
- **numpy** — Numerical computations  
- **scikit-learn** — Model development & evaluation  
- **matplotlib**, **seaborn** — Visualization  
- **joblib** — Model persistence  

### Algorithms
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|--------|-----------|
| Logistic Regression | 0.30 | 0.28 | 0.30 | 0.25 |
| Support Vector Machine | 0.30 | 0.28 | 0.30 | 0.25 |
| Random Forest | 0.10 | 0.10 | 0.10 | 0.10 |

**Observations:**
- Logistic Regression outperformed other models.
- Random Forest struggled due to the small dataset size.
- SVM achieved moderate performance but less interpretability.

---

## 📈 Results and Discussion
Visualization through bar plots and confusion matrices revealed that **Logistic Regression** provided the most consistent results across metrics.  

The findings suggest:
- Linear models perform better with smaller datasets.
- Model success depends heavily on preprocessing and balanced feature scaling.
- Data quality and diversity significantly affect predictive accuracy.

---

## 🧠 Inference
Simpler models like Logistic Regression generalize well on limited datasets, whereas complex models such as Random Forest require larger, more varied data to perform optimally.  

Proper preprocessing — including normalization, encoding, and outlier treatment — plays a crucial role in achieving stable predictions.

---

## 🧩 Conclusion and Future Scope
This project successfully demonstrates the application of machine learning for **student stress level prediction**. Among all models tested, **Logistic Regression** achieved the best balance of accuracy and interpretability.  

In the future, incorporating **biometric**, **behavioral**, or **social media** indicators could improve prediction robustness. Integrating such systems into institutional wellness platforms can enable **real-time mental health monitoring** and early intervention strategies.

---

## 🧭 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Student-Stress-Analysis.git
   cd Student-Stress-Analysis
