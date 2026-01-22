# Choice-of-Residential-Energy-Sources-LGBM

# Analysis of All-Electric vs. Dual-Fuel Housing Choice using Machine Learning

This repository contains the source code for research analyzing the factors influencing the choice between all-electric and dual-fuel housing. The analysis utilizes machine learning models and Explainable AI (XAI) to provide insights into household energy preferences.

## 🎓 Affiliation
* **Hiroshima University**, Graduate School of Advanced Science and Engineering
* **Intelligent Systems Laboratory**
* Information Science Program

---

## 📂 Repository Structure

The analysis is divided into four main scripts to ensure a clear and reproducible workflow. All results, including trained models and figures, are saved in the `result/` directory.

| File | Description |
| :--- | :--- |
| `01_eda.py` | **Exploratory Data Analysis & Mock Data Generation**. Generates a synthetic dataset for privacy protection and performs initial data visualization. |
| `02_make_model.py` | **Model Training**. Performs Nested Cross-Validation with hyperparameter tuning via Optuna for LightGBM and Logistic Regression. |
| `03_accuracy_comparison.py` | **Model Evaluation**. Calculates performance metrics (AUC, Balanced Accuracy, F1-score) and generates confusion matrices for each fold. |
| `04_data_analisis.py` | **XAI & Statistical Analysis**. Conducts SHAP analysis to interpret model decisions and performs statistical tests for feature significance. |

---

## 🛠️ Setup

### Prerequisites
* Python 3.8+
* This repository uses **Mock Data** by default to protect the privacy of the original survey participants.

### Installation
Clone this repository and install the required dependencies:
```bash
git clone [https://github.com/your-username/repository-name.git](https://github.com/your-username/repository-name.git)
cd repository-name
pip install -r requirements.txt

研究室での成果をGitHubで公開するための、マークダウン形式のREADME.mdです。

先輩の構成を参考にしつつ、今回の住宅エネルギー選択（オール電化 vs デュアルフューエル）の解析内容、モックデータの扱い、および広島大学での所属 を反映させました。これをそのままコピーしてREADME.mdとして保存してください。

Markdown
# Analysis of All-Electric vs. Dual-Fuel Housing Choice using Machine Learning

This repository contains the source code for research analyzing the factors influencing the choice between all-electric and dual-fuel housing. The analysis utilizes machine learning models and Explainable AI (XAI) to provide insights into household energy preferences.

## 🎓 Affiliation
* **Hiroshima University**, Graduate School of Advanced Science and Engineering
* **Intelligent Systems Laboratory**
* Information Science Program

---

## 📂 Repository Structure

The analysis is divided into four main scripts to ensure a clear and reproducible workflow. All results, including trained models and figures, are saved in the `result/` directory.

| File | Description |
| :--- | :--- |
| `01_eda.py` | **Exploratory Data Analysis & Mock Data Generation**. Generates a synthetic dataset for privacy protection and performs initial data visualization. |
| `02_make_model.py` | **Model Training**. Performs Nested Cross-Validation with hyperparameter tuning via Optuna for LightGBM and Logistic Regression. |
| `03_accuracy_comparison.py` | **Model Evaluation**. Calculates performance metrics (AUC, Balanced Accuracy, F1-score) and generates confusion matrices for each fold. |
| `04_data_analisis.py` | **XAI & Statistical Analysis**. Conducts SHAP analysis to interpret model decisions and performs statistical tests for feature significance. |

---

## 🛠️ Setup

### Prerequisites
* Python 3.8+
* This repository uses **Mock Data** by default to protect the privacy of the original survey participants.

### Installation
Clone this repository and install the required dependencies:
```bash
git clone [https://github.com/your-username/repository-name.git](https://github.com/your-username/repository-name.git)
cd repository-name
pip install -r requirements.txt

## 🚀 Usage
To reproduce the analysis, execute the scripts in the following order:

Data Preparation:

Bash
python 01_eda.py
Model Training:

Bash
python 02_make_model.py
Performance Evaluation:

Bash
python 03_accuracy_comparison.py
Feature Interpretation & Statistical Analysis:

Bash
python 04_data_analysis.py

## 📊 Results Output
After running the scripts, the following files will be generated in the result/ folder:

trained_models_nested_cv.pkl: Saved model objects and test data for each fold.

SHAP_summary_dot.pdf: Visualization of feature importance across the entire dataset.

SHAP_dependence_*.pdf: Analysis of specific feature interactions.

Detailed CSV/Log files for performance metrics.

## 📝 License & Citation
This project is for research purposes. If you use this code or findings in your work, please cite our corresponding research paper.
