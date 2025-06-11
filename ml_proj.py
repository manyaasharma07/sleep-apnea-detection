# Sleep Apnea Detection with Real Kaggle Dataset
# Using actual patient data instead of synthetic data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import streamlit as st
import ollama
warnings.filterwarnings('ignore')

# Add custom CSS for modern UI
st.markdown(
    """
    <style>
    .main {
        background-color: #F7F7FA;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #6C63FF 0%, #48C6EF 100%);
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1em;
        padding: 0.5em 2em;
        margin: 0.5em 0;
    }
    .stSlider > div[data-baseweb="slider"] {
        background: #E3E6F3;
        border-radius: 8px;
        padding: 0.5em;
    }
    .stDataFrame, .stTable {
        background: #fff;
        border-radius: 8px;
        font-size: 1.05em;
    }
    h1, h2, h3, h4 {
        color: #6C63FF;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-weight: 700;
    }
    .stMarkdown {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        color: #22223B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

print("Sleep Apnea Detection with Real Patient Data")
print("="*50)

# Method 1: Load data from Kaggle CSV file
def load_kaggle_sleep_data():
    """
    Load real sleep health dataset from Kaggle
    Download from: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
    """
    try:
        # Replace 'Sleep_health_and_lifestyle_dataset.csv' with your downloaded file
        df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
        print(f"✓ Loaded real dataset with {len(df)} patients")
        return df
    except FileNotFoundError:
        print("Dataset file not found. Please download from Kaggle first.")
        return None

# Method 2: Alternative - Load using Kaggle API
def load_with_kaggle_api():
    """
    Load dataset directly using Kaggle API
    Requires: pip install kaggle
    And Kaggle API credentials setup
    """
    try:
        import kaggle
        # Download dataset
        kaggle.api.dataset_download_files('uom190346a/sleep-health-and-lifestyle-dataset', 
                                         path='.', unzip=True)
        df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
        print(f"✓ Downloaded and loaded dataset with {len(df)} patients")
        return df
    except Exception as e:
        print(f"Error with Kaggle API: {e}")
        return None

# Method 3: Create sample real-world structure if dataset not available
def create_sample_real_structure():
    """
    Creates a sample dataset with the structure of real sleep datasets
    This mimics the actual Kaggle dataset format
    """
    print("Creating sample with real dataset structure...")
    np.random.seed(42)
    
    # Real dataset typically has these columns
    n_samples = 400
    
    # Simulate real patient demographics and measurements
    data = {
        'Person_ID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
        'Age': np.random.randint(25, 65, n_samples),
        'Occupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Nurse', 'Accountant'], n_samples),
        'Sleep_Duration': np.random.uniform(4.5, 9.5, n_samples),
        'Quality_of_Sleep': np.random.randint(4, 11, n_samples),
        'Physical_Activity_Level': np.random.randint(30, 91, n_samples),
        'Stress_Level': np.random.randint(3, 9, n_samples),
        'BMI_Category': np.random.choice(['Normal', 'Overweight', 'Obese'], n_samples, p=[0.4, 0.35, 0.25]),
        'Blood_Pressure': [f"{np.random.randint(110, 140)}/{np.random.randint(70, 95)}" for _ in range(n_samples)],
        'Heart_Rate': np.random.randint(65, 85, n_samples),
        'Daily_Steps': np.random.randint(3000, 12000, n_samples),
        'Sleep_Disorder': np.random.choice(['None', 'Sleep Apnea', 'Insomnia'], n_samples, p=[0.6, 0.25, 0.15])
    }
    
    df = pd.DataFrame(data)
    return df

# Try to load real data, fall back to sample if needed
df = load_kaggle_sleep_data()
if df is None:
    df = load_with_kaggle_api()
if df is None:
    df = create_sample_real_structure()
    print("Note: Using sample data structure. Download real dataset from Kaggle for actual analysis.")

# Display dataset information
print(f"\nDataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Data preprocessing for real dataset
def preprocess_real_data(df):
    """
    Preprocess the real sleep dataset for ML
    """
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.dropna()
    
    # Create binary target variable for Sleep Apnea
    if 'Sleep_Disorder' in df_processed.columns:
        df_processed['Has_Sleep_Apnea'] = (df_processed['Sleep_Disorder'] == 'Sleep Apnea').astype(int)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col not in ['Sleep_Disorder', 'Blood_Pressure']:  # Skip target and special columns
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
    
    # Handle Blood Pressure if present
    if 'Blood_Pressure' in df_processed.columns:
        # Extract systolic and diastolic values
        bp_split = df_processed['Blood_Pressure'].str.split('/', expand=True)
        df_processed['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
        df_processed['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
    
    # Handle BMI Category
    if 'BMI_Category' in df_processed.columns:
        bmi_mapping = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
        df_processed['BMI_Category_encoded'] = df_processed['BMI_Category'].map(bmi_mapping)
    
    return df_processed

# Preprocess the data
df_processed = preprocess_real_data(df)
print(f"\nProcessed dataset shape: {df_processed.shape}")

# Analyze Sleep Apnea distribution
if 'Has_Sleep_Apnea' in df_processed.columns:
    apnea_count = df_processed['Has_Sleep_Apnea'].sum()
    total_count = len(df_processed)
    print(f"\nSleep Apnea Cases: {apnea_count}/{total_count} ({apnea_count/total_count:.1%})")

# Select features for ML model
feature_columns = []
if 'Age' in df_processed.columns:
    feature_columns.append('Age')
if 'Sleep_Duration' in df_processed.columns:
    feature_columns.append('Sleep_Duration')
if 'Quality_of_Sleep' in df_processed.columns:
    feature_columns.append('Quality_of_Sleep')
if 'Physical_Activity_Level' in df_processed.columns:
    feature_columns.append('Physical_Activity_Level')
if 'Stress_Level' in df_processed.columns:
    feature_columns.append('Stress_Level')
if 'Heart_Rate' in df_processed.columns:
    feature_columns.append('Heart_Rate')
if 'Daily_Steps' in df_processed.columns:
    feature_columns.append('Daily_Steps')
if 'Systolic_BP' in df_processed.columns:
    feature_columns.append('Systolic_BP')
if 'BMI_Category_encoded' in df_processed.columns:
    feature_columns.append('BMI_Category_encoded')
if 'Gender_encoded' in df_processed.columns:
    feature_columns.append('Gender_encoded')

print(f"\nSelected features: {feature_columns}")

# Prepare data for ML if we have the target variable
if 'Has_Sleep_Apnea' in df_processed.columns and feature_columns:
    # Remove rows with missing values in selected features
    df_ml = df_processed[feature_columns + ['Has_Sleep_Apnea']].dropna()
    
    X = df_ml[feature_columns]
    y = df_ml['Has_Sleep_Apnea']
    
    print(f"\nML Dataset: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n" + "="*50)
    print("MODEL RESULTS WITH REAL DATA")
    print("="*50)
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Sleep Apnea']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance (Real Data):")
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.3f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Feature importance
    plt.subplot(2, 2, 1)
    sns.barplot(data=feature_importance, y='Feature', x='Importance')
    plt.title('Feature Importance (Real Data)')
    
    # Plot 2: Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Sleep Apnea'],
               yticklabels=['Normal', 'Sleep Apnea'])
    plt.title('Confusion Matrix')
    
    # Plot 3: Age vs Sleep Duration colored by Sleep Apnea
    if 'Age' in feature_columns and 'Sleep_Duration' in feature_columns:
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(df_ml['Age'], df_ml['Sleep_Duration'], 
                            c=df_ml['Has_Sleep_Apnea'], cmap='RdYlBu', alpha=0.6)
        plt.xlabel('Age')
        plt.ylabel('Sleep Duration (hours)')
        plt.title('Age vs Sleep Duration')
        plt.colorbar(scatter, label='Sleep Apnea (1=Yes, 0=No)')
    
    # Plot 4: Distribution of key feature by Sleep Apnea status
    if 'Quality_of_Sleep' in feature_columns:
        plt.subplot(2, 2, 4)
        df_ml[df_ml['Has_Sleep_Apnea'] == 0]['Quality_of_Sleep'].hist(alpha=0.7, label='Normal', bins=10)
        df_ml[df_ml['Has_Sleep_Apnea'] == 1]['Quality_of_Sleep'].hist(alpha=0.7, label='Sleep Apnea', bins=10)
        plt.xlabel('Quality of Sleep')
        plt.ylabel('Frequency')
        plt.title('Sleep Quality Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('real_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Analysis plots saved as 'real_data_analysis.png'")
    
    # Generate comprehensive tables like the original code
    print(f"\n" + "="*60)
    print("GENERATING COMPREHENSIVE TABLES")
    print("="*60)
    
    # Function to create and save tables as images
    def create_table_image(data, title, filename, figsize=(12, 8)):
        """Create a table visualization and save as image"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=data.values, colLabels=data.columns,
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Color the header row
        for i in range(len(data.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(data) + 1):
            for j in range(len(data.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Table saved as '{filename}'")
    
    # Table 1: Sample Data (First 20 Patients)
    sample_columns = ['Person_ID', 'Age', 'Gender', 'Sleep_Duration', 'Quality_of_Sleep', 
                     'Heart_Rate', 'Stress_Level', 'Sleep_Disorder']
    available_sample_cols = [col for col in sample_columns if col in df_processed.columns]
    
    if available_sample_cols:
        sample_data = df_processed[available_sample_cols].head(20)
        # Round numeric columns
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        sample_data[numeric_cols] = sample_data[numeric_cols].round(2)
        
        create_table_image(sample_data, 'Sample Patient Data (First 20 Records)', 
                          'real_sample_data_table.png', figsize=(14, 10))
    
    # Table 2: Comprehensive Statistics
    stats_data = []
    numeric_features = df_ml[feature_columns].select_dtypes(include=[np.number]).columns
    
    for feature in numeric_features:
        overall_mean = df_ml[feature].mean()
        overall_std = df_ml[feature].std()
        normal_mean = df_ml[df_ml['Has_Sleep_Apnea'] == 0][feature].mean()
        apnea_mean = df_ml[df_ml['Has_Sleep_Apnea'] == 1][feature].mean()
        min_val = df_ml[feature].min()
        max_val = df_ml[feature].max()
        
        stats_data.append([
            feature.replace('_', ' ').title(),
            f"{overall_mean:.2f}",
            f"{overall_std:.2f}",
            f"{min_val:.2f}",
            f"{max_val:.2f}",
            f"{normal_mean:.2f}",
            f"{apnea_mean:.2f}"
        ])
    
    stats_df = pd.DataFrame(stats_data, columns=[
        'Feature', 'Overall Mean', 'Std Dev', 'Min', 'Max', 
        'Normal Group Mean', 'Sleep Apnea Mean'
    ])
    
    create_table_image(stats_df, 'Comprehensive Feature Statistics (Real Data)', 
                      'real_statistics_table.png', figsize=(14, 8))
    
    # Table 3: Group Comparison
    normal_group = df_ml[df_ml['Has_Sleep_Apnea'] == 0]
    apnea_group = df_ml[df_ml['Has_Sleep_Apnea'] == 1]
    
    comparison_data = [
        ['Total Patients', len(normal_group), len(apnea_group), len(df_ml)],
        ['Percentage', f"{len(normal_group)/len(df_ml)*100:.1f}%", 
         f"{len(apnea_group)/len(df_ml)*100:.1f}%", "100.0%"]
    ]
    
    # Add feature comparisons
    for feature in numeric_features[:6]:  # Limit to top 6 features
        feature_name = feature.replace('_', ' ').title()
        normal_avg = normal_group[feature].mean()
        apnea_avg = apnea_group[feature].mean()
        overall_avg = df_ml[feature].mean()
        
        comparison_data.append([
            f'Avg {feature_name}',
            f"{normal_avg:.1f}",
            f"{apnea_avg:.1f}",
            f"{overall_avg:.1f}"
        ])
    
    comparison_df = pd.DataFrame(comparison_data, columns=[
        'Metric', 'Normal Group', 'Sleep Apnea Group', 'Overall'
    ])
    
    create_table_image(comparison_df, 'Sleep Apnea vs Normal Group Comparison (Real Data)', 
                      'real_comparison_table.png', figsize=(12, 8))
    
    # Table 4: Model Performance Metrics
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    performance_data = [
        ['Metric', 'Normal', 'Sleep Apnea', 'Overall'],
        ['Precision', f"{precision[0]:.3f}", f"{precision[1]:.3f}", f"{np.mean(precision):.3f}"],
        ['Recall (Sensitivity)', f"{recall[0]:.3f}", f"{recall[1]:.3f}", f"{np.mean(recall):.3f}"],
        ['F1-Score', f"{f1[0]:.3f}", f"{f1[1]:.3f}", f"{np.mean(f1):.3f}"],
        ['Support (Cases)', f"{support[0]}", f"{support[1]}", f"{sum(support)}"],
        ['Accuracy', '-', '-', f"{accuracy:.3f}"]
    ]
    
    performance_df = pd.DataFrame(performance_data[1:], columns=performance_data[0])
    
    create_table_image(performance_df, 'Model Performance Metrics (Real Data)', 
                      'real_performance_table.png', figsize=(10, 6))
    
    # Table 5: Feature Importance Rankings
    feature_importance_display = feature_importance.copy()
    feature_importance_display['Feature'] = feature_importance_display['Feature'].str.replace('_', ' ').str.title()
    feature_importance_display['Importance'] = feature_importance_display['Importance'].round(4)
    feature_importance_display['Rank'] = range(1, len(feature_importance_display) + 1)
    feature_importance_display = feature_importance_display[['Rank', 'Feature', 'Importance']]
    
    create_table_image(feature_importance_display, 'Feature Importance Rankings (Real Data)', 
                      'real_feature_importance_table.png', figsize=(10, 8))
    
    # Table 6: Clinical Prediction Examples with Real Data
    def predict_sleep_apnea_real(patient_features):
        """Predict sleep apnea for a new patient using real model"""
        try:
            prediction = model.predict([patient_features])[0]
            probability = model.predict_proba([patient_features])[0][1]
            return prediction, probability
        except:
            return 0, 0.0
    
    # Create realistic test cases based on the actual features
    test_cases = []
    if len(feature_columns) >= 3:
        # Use actual feature ranges from the data
        feature_ranges = {col: (df_ml[col].min(), df_ml[col].max()) for col in feature_columns}
        
        # Test case 1: Low risk profile
        low_risk = []
        for col in feature_columns:
            if 'Age' in col:
                low_risk.append(30)  # Young
            elif 'Sleep_Duration' in col:
                low_risk.append(8.0)  # Good sleep
            elif 'Quality' in col:
                low_risk.append(9)   # High quality
            elif 'Stress' in col:
                low_risk.append(3)   # Low stress
            elif 'Heart_Rate' in col:
                low_risk.append(65)  # Normal HR
            else:
                # Use median for other features
                low_risk.append(df_ml[col].median())
        
        # Test case 2: High risk profile
        high_risk = []
        for col in feature_columns:
            if 'Age' in col:
                high_risk.append(55)  # Older
            elif 'Sleep_Duration' in col:
                high_risk.append(5.5)  # Poor sleep
            elif 'Quality' in col:
                high_risk.append(4)   # Low quality
            elif 'Stress' in col:
                high_risk.append(8)   # High stress
            elif 'Heart_Rate' in col:
                high_risk.append(85)  # Elevated HR
            else:
                # Use 75th percentile for other features
                high_risk.append(df_ml[col].quantile(0.75))
        
        test_cases = [
            {
                'name': 'Low Risk Patient',
                'features': low_risk,
                'description': 'Young, good sleep, low stress'
            },
            {
                'name': 'High Risk Patient', 
                'features': high_risk,
                'description': 'Older, poor sleep, high stress'
            }
        ]
    
    if test_cases:
        predictions_data = []
        for case in test_cases:
            pred, prob = predict_sleep_apnea_real(case['features'])
            risk_level = "HIGH" if prob > 0.7 else "MODERATE" if prob > 0.3 else "LOW"
            
            predictions_data.append([
                case['name'],
                case['description'],
                'Sleep Apnea' if pred == 1 else 'Normal',
                f"{prob:.1%}",
                risk_level
            ])
        
        predictions_df = pd.DataFrame(predictions_data, columns=[
            'Patient Type', 'Description', 'Prediction', 'Risk Probability', 'Risk Level'
        ])
        
        create_table_image(predictions_df, 'Clinical Prediction Examples (Real Data)', 
                          'real_predictions_table.png', figsize=(14, 6))
    
    # Save processed real data
    df_processed.to_csv('processed_real_sleep_data.csv', index=False)
    print(f"\n✓ Processed real dataset saved as 'processed_real_sleep_data.csv'")
    
    # Final summary table
    summary_data = [
        ['Dataset Size', f"{len(df_processed)} patients"],
        ['Sleep Apnea Cases', f"{len(apnea_group)} ({len(apnea_group)/len(df_ml)*100:.1f}%)"],
        ['Features Used', f"{len(feature_columns)}"],
        ['Model Accuracy', f"{accuracy:.1%}"],
        ['Most Important Feature', feature_importance.iloc[0]['Feature'].replace('_', ' ').title()],
        ['Tables Generated', '6 comprehensive tables'],
        ['Plots Generated', '4 visualization plots']
    ]
    
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    
    create_table_image(summary_df, 'Project Summary (Real Data Analysis)', 
                      'real_project_summary_table.png', figsize=(10, 6))
    
    print(f"\n" + "="*60)
    print("COMPREHENSIVE TABLES GENERATED")
    print("="*60)
    print("✓ real_sample_data_table.png - Sample patient records")
    print("✓ real_statistics_table.png - Feature statistics")
    print("✓ real_comparison_table.png - Group comparisons")
    print("✓ real_performance_table.png - Model performance")
    print("✓ real_feature_importance_table.png - Feature rankings")
    print("✓ real_predictions_table.png - Clinical predictions")
    print("✓ real_project_summary_table.png - Project summary")
    print("✓ real_data_analysis.png - Analysis plots")
    print(f"✓ processed_real_sleep_data.csv - Processed dataset")

else:
    print("\nCannot proceed with ML - missing target variable or features")
    print("Please ensure you have the correct dataset with Sleep_Disorder column")

print(f"\n" + "="*50)
print("INSTRUCTIONS FOR USING REAL DATA")
print("="*50)
print("1. Download dataset from Kaggle:")
print("   https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")

st.title("Sleep Apnea Detection with Real Patient Data")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Predict"])

if page == "Data Overview":
    st.write("## Dataset Overview")
    st.dataframe(df.head())
    st.write("### Summary Statistics")
    st.dataframe(df.describe(include='all').T)

    # Distribution plots for key features
    st.write("### Feature Distributions")
    import matplotlib.pyplot as plt
    import seaborn as sns

    features_to_plot = [
        col for col in ["Age", "Sleep_Duration", "Quality_of_Sleep", "Physical_Activity_Level", "Stress_Level", "Heart_Rate", "Daily_Steps"] if col in df.columns
    ]
    for col in features_to_plot:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col.replace('_', ' ')}")
        st.pyplot(fig)

    # Sleep disorder breakdown
    if 'Sleep_Disorder' in df.columns:
        st.write("### Sleep Disorder Breakdown")
        fig, ax = plt.subplots()
        df['Sleep_Disorder'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Sleep Disorder Distribution')
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        sns.countplot(x='Sleep_Disorder', data=df, ax=ax2)
        ax2.set_title('Sleep Disorder Count')
        st.pyplot(fig2)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Pairplot for selected features
    st.write("### Pairplot of Key Features")
    import pandas as pd
    import numpy as np
    sample_df = df.sample(n=min(200, len(df)), random_state=42) if len(df) > 200 else df
    pairplot_cols = [col for col in ["Age", "Sleep_Duration", "Quality_of_Sleep", "Stress_Level"] if col in df.columns]
    if 'Sleep_Disorder' in df.columns and len(pairplot_cols) >= 2:
        fig = sns.pairplot(sample_df, vars=pairplot_cols, hue='Sleep_Disorder')
        st.pyplot(fig)
    elif len(pairplot_cols) >= 2:
        fig = sns.pairplot(sample_df, vars=pairplot_cols)
        st.pyplot(fig)

elif page == "Visualizations":
    st.write("## Interactive Data Visualizations")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    sns.set_palette('Set2')
    sns.set_style('whitegrid')

    # --- UI Controls in Expander ---
    with st.expander("🎛 Show/Hide Filters and Controls", expanded=True):
        filtered_df = df.copy()

        # Apnea label selectbox (if available)
        apnea_label = None
        if 'Sleep_Disorder' in df.columns:
            apnea_options = df['Sleep_Disorder'].unique().tolist()
            apnea_label = st.selectbox("Filter by Sleep Disorder Label", options=["All"] + apnea_options)
            if apnea_label != "All":
                filtered_df = filtered_df[filtered_df['Sleep_Disorder'] == apnea_label]

        # Sliders for continuous features
        col1, col2 = st.columns(2)
        with col1:
            if 'Sleep_Duration' in df.columns:
                min_sleep, max_sleep = float(df['Sleep_Duration'].min()), float(df['Sleep_Duration'].max())
                sleep_range = st.slider("Sleep Duration (hours)", min_sleep, max_sleep, (min_sleep, max_sleep), step=0.1)
                filtered_df = filtered_df[(filtered_df['Sleep_Duration'] >= sleep_range[0]) & (filtered_df['Sleep_Duration'] <= sleep_range[1])]
            if 'BMI_Category' in df.columns:
                bmi_options = df['BMI_Category'].unique().tolist()
                selected_bmi = st.multiselect("BMI Category", bmi_options, default=bmi_options)
                filtered_df = filtered_df[filtered_df['BMI_Category'].isin(selected_bmi)]
            if 'Age' in df.columns:
                min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
                age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
                filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
        with col2:
            if 'Quality_of_Sleep' in df.columns:
                min_qos, max_qos = int(df['Quality_of_Sleep'].min()), int(df['Quality_of_Sleep'].max())
                qos_range = st.slider("Quality of Sleep", min_qos, max_qos, (min_qos, max_qos))
                filtered_df = filtered_df[(filtered_df['Quality_of_Sleep'] >= qos_range[0]) & (filtered_df['Quality_of_Sleep'] <= qos_range[1])]
            if 'SpO2' in df.columns:
                min_spo2, max_spo2 = float(df['SpO2'].min()), float(df['SpO2'].max())
                spo2_range = st.slider("SpO₂ Range", min_spo2, max_spo2, (min_spo2, max_spo2), step=0.1)
                filtered_df = filtered_df[(filtered_df['SpO2'] >= spo2_range[0]) & (filtered_df['SpO2'] <= spo2_range[1])]

            # Checkboxes for normalization, apnea-only, smoothing
            st.markdown("### Options")
            col3, col4, col5 = st.columns(3)
            with col3:
                normalize = st.checkbox("Normalize Bar Chart", value=False)
            with col4:
                apnea_only = st.checkbox("Show Only Apnea Cases", value=False)
            with col5:
                smoothing = st.checkbox("Enable Smoothing (Line Chart)", value=False)
            if apnea_only and 'Sleep_Disorder' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Sleep_Disorder'].str.lower().str.contains('apnea')]

    st.markdown("---")

    # --- Visualizations ---
    # Defensive: If filtered_df is empty, show a warning and skip plots
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filters.")
    else:
        # Line Chart: Trend of features over time (if time-series, else by age)
        st.markdown("#### 📈 Line Chart: Feature Trend by Age")
        st.caption("Shows trend of a selected feature by age. Enable smoothing for a rolling mean.")
        line_features = [col for col in ['Sleep_Duration', 'Quality_of_Sleep', 'Physical_Activity_Level', 'Stress_Level', 'Heart_Rate', 'Daily_Steps'] if col in filtered_df.columns]
        if line_features and 'Age' in filtered_df.columns:
            line_feature = st.selectbox("Select Feature for Line Chart", line_features)
            try:
                line_data = filtered_df.groupby('Age')[line_feature].mean().sort_index()
                if smoothing:
                    line_data = line_data.rolling(window=5, min_periods=1, center=True).mean()
                fig, ax = plt.subplots()
                line_data.plot(ax=ax, color=sns.color_palette('Set2')[0])
                ax.set_ylabel(line_feature.replace('_', ' '))
                ax.set_xlabel('Age')
                ax.set_title(f'{line_feature.replace("_", " ")} Trend by Age')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Line chart error: {e}")

        # Heatmap: Correlation matrix
        st.markdown("#### 🔥 Heatmap: Feature Correlation Matrix")
        st.caption("Shows correlation between numeric features in the filtered dataset.")
        numeric_cols = filtered_df.select_dtypes(include=[float, int]).columns
        if len(numeric_cols) > 1:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap='crest', ax=ax)
                ax.set_title('Correlation Matrix')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Heatmap error: {e}")

        # Scatter Plot: User-selected axes, hue by Apnea status
        st.markdown("#### 🟢 Scatter Plot: Feature Relationship")
        st.caption("Explore the relationship between two features, colored by Apnea status. Jitter is added to help visualize overlapping points.")
        scatter_cols = [col for col in ['Age', 'Sleep_Duration', 'Quality_of_Sleep', 'Physical_Activity_Level', 'Stress_Level', 'Heart_Rate', 'Daily_Steps'] if col in filtered_df.columns]
        if len(scatter_cols) >= 2 and 'Sleep_Disorder' in filtered_df.columns:
            x_axis = st.selectbox("X Axis", scatter_cols, key='scatter_x')
            y_axis = st.selectbox("Y Axis", scatter_cols, index=1 if len(scatter_cols) > 1 else 0, key='scatter_y')
            if x_axis != y_axis:
                try:
                    # Add jitter to x and y
                    jitter_strength = 0.2  # You can adjust this value
                    x_jitter = filtered_df[x_axis] + np.random.uniform(-jitter_strength, jitter_strength, size=len(filtered_df))
                    y_jitter = filtered_df[y_axis] + np.random.uniform(-jitter_strength, jitter_strength, size=len(filtered_df))
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=x_jitter, y=y_jitter, hue=filtered_df['Sleep_Disorder'], ax=ax, palette='Set2')
                    ax.set_xlabel(x_axis.replace('_', ' '))
                    ax.set_ylabel(y_axis.replace('_', ' '))
                    ax.set_title(f'{y_axis.replace("_", " ")} vs {x_axis.replace("_", " ")}, colored by Sleep Disorder (with jitter)')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Scatter plot error: {e}")

        # Violin/KDE Plot: Distribution of a selected feature by Apnea label
        st.markdown("#### 🎻 Violin/KDE Plot: Feature Distribution by Apnea Label")
        st.caption("Shows the distribution of a selected feature by sleep disorder label.")
        if len(scatter_cols) > 0 and 'Sleep_Disorder' in filtered_df.columns:
            kde_feature = st.selectbox("Select Feature for Violin/KDE Plot", scatter_cols, key='kde')
            try:
                fig, ax = plt.subplots()
                if st.checkbox("Use Violin Plot", value=True):
                    sns.violinplot(x='Sleep_Disorder', y=kde_feature, data=filtered_df, ax=ax, palette='Set2')
                else:
                    for label in filtered_df['Sleep_Disorder'].unique():
                        sns.kdeplot(filtered_df[filtered_df['Sleep_Disorder'] == label][kde_feature], label=label, ax=ax)
                    ax.legend()
                ax.set_title(f'{kde_feature.replace("_", " ")} Distribution by Sleep Disorder')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Violin/KDE plot error: {e}")

        # Bar Chart: Feature distribution grouped by apnea status (moved to bottom)
        st.markdown("#### 📊 Bar Chart: Feature Distribution by Apnea Status")
        st.caption("Shows mean values of selected features grouped by sleep disorder label.")
        bar_features = [col for col in ['Sleep_Duration', 'Quality_of_Sleep', 'Age', 'Physical_Activity_Level', 'Stress_Level', 'Heart_Rate', 'Daily_Steps'] if col in filtered_df.columns]
        if bar_features and 'Sleep_Disorder' in filtered_df.columns:
            bar_feature = st.selectbox("Select Feature for Bar Chart", bar_features)
            try:
                bar_data = filtered_df.groupby('Sleep_Disorder')[bar_feature].mean()
                if normalize and bar_data.sum() != 0:
                    bar_data = bar_data / bar_data.sum()
                fig, ax = plt.subplots()
                bar_data.plot(kind='bar', ax=ax, color=sns.color_palette('Set2'))
                ax.set_ylabel('Normalized Mean' if normalize else 'Mean')
                ax.set_xlabel('Sleep Disorder')
                ax.set_title(f'{bar_feature.replace("_", " ")} by Sleep Disorder')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Bar chart error: {e}")

elif page == "Predict":
    st.write("## Predict Sleep Apnea")

    st.write("Enter your details to analyze your sleep health:")

    # Collect user input for each feature
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1=Poor, 10=Excellent)", 1, 10, 7)
    physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 180, 60)
    stress_level = st.slider("Stress Level (1=Low, 10=High)", 1, 10, 5)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=1000, max_value=30000, value=7000)
    systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    # Encode categorical variables as in your preprocessing
    bmi_mapping = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
    gender_mapping = {'Male': 1, 'Female': 0}
    bmi_encoded = bmi_mapping[bmi_category]
    gender_encoded = gender_mapping[gender]

    # Prepare the feature vector in the same order as your model
    input_features = []
    if 'Age' in feature_columns:
        input_features.append(age)
    if 'Sleep_Duration' in feature_columns:
        input_features.append(sleep_duration)
    if 'Quality_of_Sleep' in feature_columns:
        input_features.append(quality_of_sleep)
    if 'Physical_Activity_Level' in feature_columns:
        input_features.append(physical_activity)
    if 'Stress_Level' in feature_columns:
        input_features.append(stress_level)
    if 'Heart_Rate' in feature_columns:
        input_features.append(heart_rate)
    if 'Daily_Steps' in feature_columns:
        input_features.append(daily_steps)
    if 'Systolic_BP' in feature_columns:
        input_features.append(systolic_bp)
    if 'BMI_Category_encoded' in feature_columns:
        input_features.append(bmi_encoded)
    if 'Gender_encoded' in feature_columns:
        input_features.append(gender_encoded)

    if st.button("Analyze My Sleep Health"):
        try:
            if 'model' not in globals() or 'feature_columns' not in globals():
                raise Exception("Model or feature_columns not defined. Please check your data and model training section.")
            pred = model.predict([input_features])[0]
            prob = model.predict_proba([input_features])[0][1]

            if pred == 1:
                st.error(f"⚠️ High risk of Sleep Apnea! (Probability: {prob:.1%})")
                st.write("**AI Analysis:** Your profile suggests a higher risk for sleep apnea. Consider consulting a healthcare professional. Improving sleep quality, reducing stress, and increasing physical activity may help.")
            else:
                st.success(f"✅ Low risk of Sleep Apnea. (Probability: {prob:.1%})")
                st.write("**AI Analysis:** Your sleep health profile appears normal. Maintain healthy habits for continued well-being.")

            # Show a summary table
            st.write("### Your Input Summary")
            st.table({
                "Age": age,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_of_sleep,
                "Physical Activity": physical_activity,
                "Stress Level": stress_level,
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps,
                "Systolic BP": systolic_bp,
                "BMI Category": bmi_category,
                "Gender": gender
            })

            # Recommendations section
            st.write("### Personalized Recommendations to Reduce Sleep Apnea Risk")
            recs = []
            if sleep_duration < 7:
                recs.append("- Try to increase your sleep duration to at least 7 hours per night.")
            if quality_of_sleep < 7:
                recs.append("- Improve your sleep quality by maintaining a regular sleep schedule and creating a restful environment.")
            if stress_level > 6:
                recs.append("- Practice stress management techniques such as meditation, deep breathing, or yoga.")
            if physical_activity < 30:
                recs.append("- Increase your daily physical activity. Aim for at least 30 minutes of moderate exercise.")
            if bmi_category == "Overweight" or bmi_category == "Obese":
                recs.append("- Work towards a healthy weight through balanced diet and exercise.")
            if heart_rate > 85:
                recs.append("- Monitor your heart health and consult a doctor if your resting heart rate is consistently high.")
            if not recs:
                recs.append("- Keep up your healthy habits! Your profile shows good sleep health practices.")
            for rec in recs:
                st.write(rec)

            # ELI5/Feature Importance Explanation
            try:
                import eli5
                st.write("### Top Model Features (ELI5)")
                importances = model.feature_importances_
                top_feats = sorted(zip(feature_columns, importances), key=lambda x: -x[1])[:5]
                for feat, imp in top_feats:
                    st.write(f"- **{feat.replace('_', ' ').title()}**: {imp:.3f}")
            except Exception as e:
                st.info(f"(Feature importance explanation unavailable: {e})")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # --- Llama 3 AI Recommendations Section (Local via Ollama) ---
    def get_llama3_recommendations(profile):
        prompt = (
            f"You are a health and wellness coach. "
            f"Given this profile: {profile}, first provide a brief analysis of the user's sleep health and risk factors for sleep apnea. "
            f"Then, give 3-5 specific, practical lifestyle recommendations to reduce sleep apnea risk, tailored to the analysis. "
            f"Include advice on eating habits, exercise, and sleep hygiene. Be concise and actionable. Add emojis, keep it short and doable, and use bullet points for the recommendations."
        )
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']

    if st.button("Get AI-Powered Lifestyle Recommendations (Llama 3)"):
        profile = f"""
        Age: {age}
        BMI: {bmi_category}
        Sleep Duration: {sleep_duration}
        Quality of Sleep: {quality_of_sleep}
        Physical Activity: {physical_activity}
        Stress Level: {stress_level}
        Heart Rate: {heart_rate}
        """
        try:
            with st.spinner("Getting Llama 3 recommendations..."):
                advice = get_llama3_recommendations(profile)
            st.write("### AI-Powered Lifestyle Recommendations (Llama 3)")
            st.write(advice)
        except Exception as e:
            st.error(f"Llama 3 error: {e}")


