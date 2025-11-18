"""
Healthcare Dataset Analysis
Novintix Assessment - Complete Analysis Pipeline

Dataset: Healthcare Dataset from Kaggle
Tasks: EDA, Supervised Learning, Unsupervised Learning, AI Recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("HEALTHCARE DATASET ANALYSIS - NOVINTIX ASSESSMENT")
print("="*80)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n[1] Loading Dataset...")

# Note: Download the dataset from Kaggle first
# https://www.kaggle.com/datasets/prasad22/healthcare-dataset
# Place it in the same directory as this script

try:
    df = pd.read_csv('healthcare_dataset.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError:
    print("\n❌ Error: 'healthcare_dataset.csv' not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/prasad22/healthcare-dataset")
    print("and place it in the same directory as this script.")
    exit(1)

# Display basic info
print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================================
# TASK 1 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("TASK 1: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Create output directory for plots
import os
os.makedirs('output_plots', exist_ok=True)

# 1.1 Distribution Analysis: Age, Billing Amount, Room Number
print("\n[1.1] Analyzing distributions of Age, Billing Amount, and Room Number...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age Distribution
axes[0].hist(df['Age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Age', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {df["Age"].mean():.1f}')
axes[0].legend()

# Billing Amount Distribution
axes[1].hist(df['Billing Amount'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_title('Distribution of Billing Amount', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Billing Amount ($)')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['Billing Amount'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${df["Billing Amount"].mean():.2f}')
axes[1].legend()

# Room Number Distribution
axes[2].hist(df['Room Number'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
axes[2].set_title('Distribution of Room Number', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Room Number')
axes[2].set_ylabel('Frequency')
axes[2].axvline(df['Room Number'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["Room Number"].mean():.1f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('output_plots/task1_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output_plots/task1_distributions.png")
plt.close()

# Statistical Summary
print("\nStatistical Summary:")
print(df[['Age', 'Billing Amount', 'Room Number']].describe())

# 1.2 Frequency Visualizations: Medical Condition, Admission Type, Medication
print("\n[1.2] Visualizing frequencies of Medical Condition, Admission Type, and Medication...")

fig, axes = plt.subplots(3, 1, figsize=(14, 15))

# Medical Condition
medical_condition_counts = df['Medical Condition'].value_counts()
axes[0].barh(medical_condition_counts.index, medical_condition_counts.values, color='steelblue')
axes[0].set_title('Frequency of Medical Conditions', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Medical Condition')
for i, v in enumerate(medical_condition_counts.values):
    axes[0].text(v + 50, i, str(v), va='center')

# Admission Type
admission_type_counts = df['Admission Type'].value_counts()
axes[1].barh(admission_type_counts.index, admission_type_counts.values, color='coral')
axes[1].set_title('Frequency of Admission Types', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Admission Type')
for i, v in enumerate(admission_type_counts.values):
    axes[1].text(v + 50, i, str(v), va='center')

# Medication
medication_counts = df['Medication'].value_counts().head(15)  # Top 15 medications
axes[2].barh(medication_counts.index, medication_counts.values, color='mediumseagreen')
axes[2].set_title('Frequency of Top 15 Medications', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Count')
axes[2].set_ylabel('Medication')
for i, v in enumerate(medication_counts.values):
    axes[2].text(v + 20, i, str(v), va='center')

plt.tight_layout()
plt.savefig('output_plots/task1_frequencies.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output_plots/task1_frequencies.png")
plt.close()

# ============================================================================
# TASK 2 - SUPERVISED LEARNING (Predict Test Results)
# ============================================================================
print("\n" + "="*80)
print("TASK 2: SUPERVISED LEARNING - PREDICTING TEST RESULTS")
print("="*80)

print("\n[2.1] Preparing dataset for prediction...")

# Create a copy for modeling
df_model = df.copy()

# Engineer additional features
df_model['Age_Group'] = pd.cut(df_model['Age'], bins=[0, 18, 35, 50, 65, 100], 
                                labels=['Child', 'Young', 'Middle', 'Senior', 'Elderly'])
df_model['Billing_Category'] = pd.cut(df_model['Billing Amount'], bins=5, 
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Select features for prediction - expanded feature set
feature_columns = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 
                   'Admission Type', 'Billing Amount', 'Medication', 
                   'Room Number', 'Age_Group', 'Billing_Category']

# Prepare data
X = df_model[feature_columns].copy()
y = df_model['Test Results'].copy()

print(f"Target variable (Test Results) distribution:\n{y.value_counts()}")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 
                       'Medication', 'Age_Group', 'Billing_Category']

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print(f"\n✓ Features prepared: {X.shape}")
print(f"✓ Target encoded: {len(np.unique(y_encoded))} classes")

# Split dataset
print("\n[2.2] Splitting dataset into training and testing sets (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train improved model with better hyperparameters
print("\n[2.3] Training Optimized Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,           # Increased from 100
    max_depth=20,               # Increased from 10
    min_samples_split=2,        # Decreased from 5
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    class_weight='balanced'     # Handle class imbalance
)
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully!")

# Make predictions
print("\n[2.4] Making predictions on test data...")
y_pred = model.predict(X_test_scaled)
y_pred_labels = le_target.inverse_transform(y_pred)
y_test_labels = le_target.inverse_transform(y_test)

# Evaluate performance
print("\n[2.5] Model Performance Evaluation:")
print("="*60)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred_labels)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_target.classes_, 
            yticklabels=le_target.classes_)
plt.title('Confusion Matrix - Test Results Prediction', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('output_plots/task2_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: output_plots/task2_confusion_matrix.png")
plt.close()

# Display Predicted vs Actual for sample test data
print("\n[2.6] Predicted vs Actual Values (First 20 test samples):")
print("="*60)
comparison_df = pd.DataFrame({
    'Actual': y_test_labels[:20],
    'Predicted': y_pred_labels[:20],
    'Match': y_test_labels[:20] == y_pred_labels[:20]
})
print(comparison_df.to_string(index=True))

# Save full predictions to CSV
predictions_df = pd.DataFrame({
    'Actual_Test_Result': y_test_labels,
    'Predicted_Test_Result': y_pred_labels,
    'Correct': y_test_labels == y_pred_labels
})
predictions_df.to_csv('output_plots/task2_predictions.csv', index=False)
print(f"\n✓ Full predictions saved to: output_plots/task2_predictions.csv")

# ============================================================================
# TASK 3 - UNSUPERVISED LEARNING (Anomaly Detection in Billing Amounts)
# ============================================================================
print("\n" + "="*80)
print("TASK 3: UNSUPERVISED LEARNING - ANOMALY DETECTION IN BILLING AMOUNTS")
print("="*80)

print("\n[3.1] Performing anomaly detection on Billing Amounts...")

# Prepare data for anomaly detection
billing_data = df[['Billing Amount']].values

# Use Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% expected anomalies
anomaly_labels = iso_forest.fit_predict(billing_data)

# -1 for anomalies, 1 for normal
df['Anomaly'] = anomaly_labels
df['Is_Anomaly'] = df['Anomaly'] == -1

# Statistics
num_anomalies = df['Is_Anomaly'].sum()
print(f"✓ Total records analyzed: {len(df)}")
print(f"✓ Anomalies detected: {num_anomalies} ({num_anomalies/len(df)*100:.2f}%)")
print(f"✓ Normal cases: {len(df) - num_anomalies} ({(len(df)-num_anomalies)/len(df)*100:.2f}%)")

# Analyze anomalies
print("\n[3.2] Anomaly Statistics:")
print("="*60)
print(f"Normal Billing Amount - Mean: ${df[~df['Is_Anomaly']]['Billing Amount'].mean():.2f}")
print(f"Normal Billing Amount - Median: ${df[~df['Is_Anomaly']]['Billing Amount'].median():.2f}")
print(f"Normal Billing Amount - Std: ${df[~df['Is_Anomaly']]['Billing Amount'].std():.2f}")
print()
print(f"Anomalous Billing Amount - Mean: ${df[df['Is_Anomaly']]['Billing Amount'].mean():.2f}")
print(f"Anomalous Billing Amount - Median: ${df[df['Is_Anomaly']]['Billing Amount'].median():.2f}")
print(f"Anomalous Billing Amount - Std: ${df[df['Is_Anomaly']]['Billing Amount'].std():.2f}")

# Visualize anomalies
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Scatter plot
axes[0].scatter(df[~df['Is_Anomaly']].index, df[~df['Is_Anomaly']]['Billing Amount'], 
                c='blue', label='Normal', alpha=0.5, s=20)
axes[0].scatter(df[df['Is_Anomaly']].index, df[df['Is_Anomaly']]['Billing Amount'], 
                c='red', label='Anomaly', alpha=0.8, s=50, marker='x')
axes[0].set_title('Anomaly Detection in Billing Amounts', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Record Index')
axes[0].set_ylabel('Billing Amount ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot comparison
billing_comparison = pd.DataFrame({
    'Billing Amount': df['Billing Amount'],
    'Category': df['Is_Anomaly'].map({True: 'Anomaly', False: 'Normal'})
})
sns.boxplot(x='Category', y='Billing Amount', data=billing_comparison, ax=axes[1])
axes[1].set_title('Billing Amount Distribution: Normal vs Anomaly', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Billing Amount ($)')

plt.tight_layout()
plt.savefig('output_plots/task3_anomaly_detection.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: output_plots/task3_anomaly_detection.png")
plt.close()

# Display sample anomalies
print("\n[3.3] Sample Detected Anomalies (First 10):")
print("="*60)
anomaly_samples = df[df['Is_Anomaly']][['Name', 'Age', 'Medical Condition', 'Billing Amount', 
                                          'Admission Type', 'Hospital']].head(10)
print(anomaly_samples.to_string(index=False))

# Save all anomalies to CSV
anomalies_df = df[df['Is_Anomaly']][['Name', 'Age', 'Gender', 'Medical Condition', 
                                       'Billing Amount', 'Admission Type', 'Hospital', 
                                       'Insurance Provider']].copy()
anomalies_df = anomalies_df.sort_values('Billing Amount', ascending=False)
anomalies_df.to_csv('output_plots/task3_detected_anomalies.csv', index=False)
print(f"\n✓ All anomalies saved to: output_plots/task3_detected_anomalies.csv")

# Interpretation
print("\n[3.4] Anomaly Interpretation:")
print("="*60)
print("""
The anomaly detection analysis reveals the following insights:

1. HIGH BILLING ANOMALIES:
   - Unusually expensive cases likely due to:
     * Complex medical procedures or surgeries
     * Extended hospital stays
     * Rare or expensive medications
     * Complications requiring intensive care
   
2. LOW BILLING ANOMALIES:
   - Unusually low billing cases may indicate:
     * Brief consultations or check-ups
     * Insurance coverage edge cases
     * Administrative or billing errors
     * Preventive care visits

3. RECOMMENDATIONS:
   - High anomalies should be reviewed for:
     * Proper insurance claims processing
     * Justification of treatment costs
     * Potential fraud detection
   
   - Low anomalies should be checked for:
     * Billing completeness
     * Coding accuracy
     * Missing charges

4. PATTERNS OBSERVED:
""")

# Analyze anomaly patterns by medical condition
anomaly_by_condition = df[df['Is_Anomaly']]['Medical Condition'].value_counts()
print(f"   Top medical conditions in anomalies:\n{anomaly_by_condition.head()}")

print(f"\n   Anomaly billing range: ${df[df['Is_Anomaly']]['Billing Amount'].min():.2f} - ${df[df['Is_Anomaly']]['Billing Amount'].max():.2f}")
print(f"   Normal billing range: ${df[~df['Is_Anomaly']]['Billing Amount'].min():.2f} - ${df[~df['Is_Anomaly']]['Billing Amount'].max():.2f}")

# ============================================================================
# TASK 4 - AI TASK (LLM-Based Doctor Recommendation Generator)
# ============================================================================
print("\n" + "="*80)
print("TASK 4: AI DOCTOR RECOMMENDATION GENERATOR")
print("="*80)

print("\n[4.1] Generating AI Doctor Recommendations...")

def generate_doctor_recommendation(age, medical_condition, medication, test_result):
    """
    Generate a doctor-style recommendation based on patient attributes.
    This simulates an LLM-based recommendation system.
    """
    
    recommendation = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     MEDICAL RECOMMENDATION REPORT                          ║
╚════════════════════════════════════════════════════════════════════════════╝

PATIENT PROFILE:
├─ Age: {age} years
├─ Medical Condition: {medical_condition}
├─ Current Medication: {medication}
└─ Test Result: {test_result}

CLINICAL ASSESSMENT:
"""
    
    # Generate recommendations based on test results and condition
    if test_result.lower() == 'normal':
        recommendation += """
The patient's test results are within normal parameters, which is encouraging.
This indicates that the current treatment plan is effective.

"""
    elif test_result.lower() == 'abnormal':
        recommendation += """
⚠ ATTENTION: Test results show abnormal findings that require further evaluation.
Immediate follow-up is recommended to address these concerns.

"""
    else:  # Inconclusive
        recommendation += """
⚠ NOTE: Test results are inconclusive and may require additional testing.
Further diagnostic procedures are recommended for accurate assessment.

"""
    
    # Age-based recommendations
    if age < 18:
        recommendation += """PEDIATRIC CONSIDERATIONS:
- Monitor growth and development closely
- Ensure age-appropriate medication dosage
- Consider family history and genetic factors
"""
    elif age >= 65:
        recommendation += """GERIATRIC CONSIDERATIONS:
- Monitor for age-related complications
- Regular assessment of medication side effects
- Consider comorbidities and polypharmacy risks
- Ensure adequate nutrition and hydration
"""
    
    # Condition-specific recommendations
    condition_recommendations = {
        'Diabetes': """
DIABETES MANAGEMENT RECOMMENDATIONS:
✓ Continue monitoring blood glucose levels regularly (fasting and post-prandial)
✓ Maintain a balanced diet with controlled carbohydrate intake
✓ Engage in regular physical activity (30 minutes daily, as tolerated)
✓ Schedule quarterly HbA1c tests to track long-term glucose control
✓ Monitor for diabetic complications (retinopathy, nephropathy, neuropathy)
✓ Keep feet care routine and watch for any wounds or infections
""",
        'Hypertension': """
HYPERTENSION MANAGEMENT RECOMMENDATIONS:
✓ Continue prescribed antihypertensive medication as directed
✓ Monitor blood pressure daily at home (keep a log)
✓ Reduce sodium intake (<2300mg per day)
✓ Maintain healthy weight through diet and exercise
✓ Limit alcohol consumption and avoid smoking
✓ Manage stress through relaxation techniques
✓ Schedule regular follow-ups for medication adjustment if needed
""",
        'Asthma': """
ASTHMA MANAGEMENT RECOMMENDATIONS:
✓ Continue controller medications as prescribed
✓ Keep rescue inhaler accessible at all times
✓ Monitor peak flow readings regularly
✓ Identify and avoid asthma triggers (allergens, irritants)
✓ Follow asthma action plan, especially during exacerbations
✓ Get annual flu vaccination
✓ Schedule pulmonary function tests as recommended
""",
        'Arthritis': """
ARTHRITIS MANAGEMENT RECOMMENDATIONS:
✓ Continue anti-inflammatory medications as prescribed
✓ Engage in low-impact exercises (swimming, walking, yoga)
✓ Apply heat or cold therapy for pain relief
✓ Maintain healthy weight to reduce joint stress
✓ Consider physical therapy for joint mobility
✓ Use assistive devices if needed for daily activities
✓ Monitor for medication side effects (GI issues with NSAIDs)
""",
        'Cancer': """
CANCER CARE RECOMMENDATIONS:
✓ Adhere strictly to prescribed treatment protocol
✓ Attend all scheduled chemotherapy/radiation sessions
✓ Monitor for side effects and report immediately
✓ Maintain optimal nutrition (high-protein, high-calorie diet)
✓ Stay hydrated and manage fatigue with adequate rest
✓ Seek psychological support (counseling, support groups)
✓ Regular imaging and tumor marker monitoring
✓ Coordinate care with oncology team for comprehensive management
""",
        'Obesity': """
WEIGHT MANAGEMENT RECOMMENDATIONS:
✓ Develop a structured weight loss plan (target 1-2 lbs/week)
✓ Follow a balanced, calorie-controlled diet
✓ Increase physical activity gradually (target 150 min/week)
✓ Keep food and exercise diary for accountability
✓ Address emotional eating and seek behavioral therapy if needed
✓ Consider consultation with nutritionist/dietitian
✓ Monitor for obesity-related complications (diabetes, hypertension)
✓ Set realistic, achievable goals with regular progress checks
"""
    }
    
    # Add condition-specific recommendation if available
    for condition, advice in condition_recommendations.items():
        if condition.lower() in medical_condition.lower():
            recommendation += advice
            break
    else:
        # Generic recommendation if condition not in predefined list
        recommendation += f"""
GENERAL MEDICAL RECOMMENDATIONS:
✓ Continue current medication ({medication}) as prescribed
✓ Monitor symptoms and report any changes to healthcare provider
✓ Maintain healthy lifestyle habits (diet, exercise, sleep)
✓ Schedule regular follow-up appointments
✓ Stay compliant with prescribed treatment regimen
"""
    
    # Medication-specific notes
    recommendation += f"""
MEDICATION GUIDANCE:
- Current Prescription: {medication}
- Take medication as directed, do not skip doses
- Report any side effects or adverse reactions immediately
- Do not discontinue medication without consulting your doctor
- Keep track of refills and maintain adequate supply
"""
    
    # General health advice
    recommendation += """
LIFESTYLE & PREVENTIVE CARE:
├─ Nutrition: Eat a balanced diet rich in fruits, vegetables, and whole grains
├─ Hydration: Drink adequate water (8-10 glasses daily)
├─ Exercise: Regular physical activity based on your ability
├─ Sleep: Aim for 7-9 hours of quality sleep per night
├─ Stress: Practice stress management techniques
└─ Preventive: Stay up-to-date with vaccinations and screenings

FOLLOW-UP PLAN:
"""
    
    if test_result.lower() == 'abnormal':
        recommendation += """
⚠ URGENT: Schedule follow-up within 1-2 weeks
- Repeat relevant tests to confirm findings
- Consider specialist referral if needed
- Possible medication adjustment
"""
    elif test_result.lower() == 'inconclusive':
        recommendation += """
- Schedule follow-up within 2-4 weeks
- Additional diagnostic tests may be ordered
- Continue current treatment plan in the interim
"""
    else:
        recommendation += """
- Routine follow-up in 3-6 months
- Continue current management plan
- Report any new symptoms before scheduled visit
"""
    
    recommendation += """
EMERGENCY CONTACT:
Contact your healthcare provider or visit emergency room if you experience:
• Severe chest pain or difficulty breathing
• Sudden severe headache or vision changes
• Signs of allergic reaction to medication
• Uncontrolled bleeding or severe injury
• Any symptom that causes significant concern

═══════════════════════════════════════════════════════════════════════════

This recommendation is generated based on the patient's current medical profile.
Always consult with a licensed healthcare provider for personalized medical advice.

Generated: November 18, 2025
═══════════════════════════════════════════════════════════════════════════
"""
    
    return recommendation

# Generate sample recommendation using first test patient
sample_idx = 0
sample_patient = {
    'age': int(X_test.iloc[sample_idx]['Age']),
    'medical_condition': le_target.inverse_transform([y_test[sample_idx]])[0],  # Use actual for demo
    'medication': df.iloc[X_test.index[sample_idx]]['Medication'],
    'test_result': y_pred_labels[sample_idx]  # Use predicted result
}

print("\n[4.2] Sample AI-Generated Doctor Recommendation:")
print("="*80)

recommendation_output = generate_doctor_recommendation(
    age=sample_patient['age'],
    medical_condition=df.iloc[X_test.index[sample_idx]]['Medical Condition'],
    medication=sample_patient['medication'],
    test_result=sample_patient['test_result']
)

print(recommendation_output)

# Save recommendation to file
with open('output_plots/task4_sample_recommendation.txt', 'w', encoding='utf-8') as f:
    f.write(recommendation_output)
print("\n✓ Saved: output_plots/task4_sample_recommendation.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*80)

print("""
All tasks completed successfully! ✓

FILES GENERATED:
├── output_plots/
│   ├── task1_distributions.png          (EDA: Age, Billing, Room distributions)
│   ├── task1_frequencies.png            (EDA: Condition, Admission, Medication frequencies)
│   ├── task2_confusion_matrix.png       (Supervised: Model performance)
│   ├── task2_predictions.csv            (Supervised: Predicted vs actual values)
│   ├── task3_anomaly_detection.png      (Unsupervised: Billing anomalies)
│   ├── task3_detected_anomalies.csv     (Unsupervised: Flagged anomalous records)
│   └── task4_sample_recommendation.txt  (AI: Doctor recommendation sample)

TASK COMPLETION STATUS:
✓ Task 1: EDA - Distributions and frequency analyses complete
✓ Task 2: Supervised Learning - Model trained with {accuracy*100:.2f}% accuracy
✓ Task 3: Unsupervised Learning - {num_anomalies} billing anomalies detected
✓ Task 4: AI Recommendations - Sample doctor recommendation generated
""")

print("="*80)
print("Thank you for using the Healthcare Dataset Analysis System!")
print("="*80)
