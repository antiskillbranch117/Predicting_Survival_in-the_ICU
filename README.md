# Predicting_Survival_in-the_ICU
This project applies machine learning to real-world healthcare data from ICU patients (Beth Israel Deaconess Medical Center, PhysioNet). The goal is to build classification models that can predict in-hospital mortality and 30-day mortality using features extracted from patient records.

## Overview  
This project investigates how machine learning can be applied to healthcare by predicting in-hospital and 30-day mortality of ICU patients.  
The dataset is drawn from the **PhysioNet ICU database**, which contains around **12,000 admissions** from the Beth Israel Deaconess Medical Center.  
Each patient record includes timestamped clinical measurements taken during the first 48 hours of ICU stay, and the task is to predict survival outcomes.  
The last 2,000 patients are reserved as a hidden test set to evaluate model generalization.  

---

## Data Preprocessing  
- **Static variables**: kept as provided.  
- **Time-varying variables**: summarized using maximum value across 48 hours.  
- **Missing values**: imputed with `NaN` (later handled by mean).  
- **Normalization**: all features scaled to `[0,1]`.  
- **Categorical variables**: one-hot encoded.
  
---

## Feature Extraction  
In addition to the main project, I also focused on improving feature extraction:  

1. **Numerical variables**: kept as floats; missing data imputed with `NaN`, replaced with mean later.  
2. **Categorical variables**: hot encoded.
3. **ICU categories**: extended feature vectors to cover each ICU type.  
4. **Time-varying variables**: summarized using **max, min, mean, std**.  
5. **Subset features**: split the 48-hour window into two 24-hour windows to capture temporal trends.  
   
---

## Models  
### Logistic Regression  
- Explored **ℓ1 (lasso)** and **ℓ2 (ridge)** penalties.  
- Hyperparameter tuning with **cross-validation**.  
- **Weighted logistic regression** to handle class imbalance.
- **Hyperparameter tuning**: searched across multiple `C` values and both ℓ1 and ℓ2 penalties.   

### Kernel Ridge Regression (RBF Kernel)  
- Captured **non-linear patterns** beyond linear models.  
- Compared trade-offs between interpretability and flexibility.  

---

## Evaluation  
- Metrics: **Accuracy, Precision, Recall, F1-score, AUROC, Average Precision**.  
- **Bootstrapping** to compute 95% confidence intervals.  
- Plotted **ROC/PR curves** to analyze sensitivity–specificity trade-offs.  
- Key predictors:  
- **Age** — important demographic predictor, older patients tend to have worse outcomes.  
- **Gender** — potential biological differences in survival.  
- **ICUType** — type of ICU (e.g., medical, surgical) reflects baseline severity and case mix.  
- **Weight & Height** — baseline physiological status.  
- **Vital signs (HR, MAP, RespRate, Temp, SaO2, etc.)** — indicators of acute physiological state.  
- **Laboratory values (Creatinine, BUN, Lactate, Glucose, WBC, Platelets, etc.)** — capture organ function and metabolic status, key for survival prediction.  
- **Other markers (pH, PaO2, PaCO2, Troponins, etc.)** — reflect critical illness severity.  

 

**Results on challenge data**:  
- AUROC = **0.8571**  
- F1-score = **0.5163**  
- Best config: `C=1`, penalty = **ℓ1**  

---

## Tech Stack  
- **Python 3.13.3**  
- **NumPy, Pandas** for data handling  
- **scikit-learn** for modeling & evaluation  
- **Matplotlib** for visualization  

---

## Repository Structure  
