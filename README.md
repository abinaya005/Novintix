# Healthcare Dataset Analysis - Novintix Assessment

Complete analysis of healthcare dataset implementing EDA, supervised learning, unsupervised learning, and AI-generated recommendations.

## Dataset

**Source**: [Healthcare Dataset on Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)

**Columns**: Name, Age, Gender, Blood Type, Medical Condition, Date of Admission, Doctor, Hospital, Insurance Provider, Billing Amount, Room Number, Admission Type, Discharge Date, Medication, Test Results

## Setup & Execution

### 1. Download Dataset
Visit: https://www.kaggle.com/datasets/prasad22/healthcare-dataset  
Save as `healthcare_dataset.csv` in this directory

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Analysis
```powershell
python healthcare_analysis.py
```

## Tasks Implemented

### Task 1: Exploratory Data Analysis (EDA)
- Analyze distributions of Age, Billing Amount, and Room Number
- Visualize frequency of Medical Condition, Admission Type, and Medication
- **Outputs**: 
  - `task1_distributions.png`
  - `task1_frequencies.png`

### Task 2: Supervised Learning
- Prepare dataset for predicting Test Results
- Split into training and testing sets (80-20)
- Train Random Forest Classifier
- Evaluate performance using appropriate metrics
- Display predicted vs actual values for test data
- **Outputs**:
  - `task2_confusion_matrix.png`
  - `task2_predictions.csv`

### Task 3: Unsupervised Learning - Anomaly Detection
- Identify unusually high or low Billing Amount values
- Detect entries deviating from expected billing patterns
- Mark detected anomalies for review
- Provide interpretation of anomalies
- **Outputs**:
  - `task3_anomaly_detection.png`
  - `task3_detected_anomalies.csv`

### Task 4: AI Doctor Recommendation Generator
- Pass patient's predicted test result and key attributes
- Generate AI-written doctor-style recommendation
- Include health advice based on predicted result
- Produce one sample output
- **Output**:
  - `task4_sample_recommendation.txt`

## Output Files

All files in `output_plots/` directory:
- `task1_distributions.png` - Age, Billing Amount, Room Number distributions
- `task1_frequencies.png` - Medical conditions, admission types, medications
- `task2_confusion_matrix.png` - Model performance visualization
- `task2_predictions.csv` - Predicted vs actual test results
- `task3_anomaly_detection.png` - Billing anomaly visualization
- `task3_detected_anomalies.csv` - Flagged anomalous records
- `task4_sample_recommendation.txt` - AI doctor recommendation sample

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn

Date: November 18, 2025
