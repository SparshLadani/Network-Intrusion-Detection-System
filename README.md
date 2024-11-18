---

# Network Intrusion Detection System (NIDS)

This project implements a Network Intrusion Detection System (NIDS) using Python and machine learning techniques. It utilizes a dataset from Kaggle to identify malicious activities within network traffic. The primary objective is to classify network activity as normal or malicious based on features in network traffic data. Please Note that this project is currently under development.

## Project Structure

- **Data Source**: [Network Traffic Data for Malicious Activity Detection](https://www.kaggle.com/datasets/advaitnmenon/network-traffic-data-malicious-activity-detection)
- **Framework**: Jupyter Notebook
- **Main Libraries**:
  - `pandas`, `numpy`: Data manipulation and analysis
  - `matplotlib`: Data visualization
  - `scikit-learn`: Machine learning models and metrics

---

## Dataset

The dataset used for this project is sourced from Kaggle and contains various attributes related to network traffic, such as `Time`, `Source`, `Destination`, `Protocol`, and more. Each record in the dataset is labeled to indicate if it is a malicious activity (`bad_packet`).

### Key Features:

- **Time**: Timestamp of the packet capture.
- **Source**: The source IP address or identifier of the packet.
- **Destination**: The destination IP address or identifier.
- **Protocol**: Protocol type (e.g., ARP, TCP).
- **Length**: Length of the packet.
- **bad_packet**: Label indicating if the packet is malicious (1) or normal (0).

---

## Project Workflow

### 1. **Import Libraries**

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.svm import SVC
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.compose import ColumnTransformer
   ```

### 2. **Data Loading and Preprocessing**

   - Load the dataset, dropping unnecessary columns (`Source Port`, `Destination Port`) to simplify analysis.
   - Basic data cleaning and feature engineering, such as one-hot encoding categorical variables.

   ```python
   path = "/path/to/network_anomaly.csv"
   data = pd.read_csv(path, low_memory=False)
   data = data.drop(columns=['Source Port', 'Destination Port'])
   ```

### 3. **Exploratory Data Analysis (EDA)**

   - Visualize distribution of malicious and non-malicious traffic.
   - Generate histograms and distribution plots to understand network traffic patterns.

### 4. **Feature Engineering**

   - Categorical columns (e.g., `Protocol`) are transformed using OneHotEncoder to prepare them for model input.

   ```python
   column_transformer = ColumnTransformer(
       transformers=[
           ('encoder', OneHotEncoder(), ['Protocol'])
       ], remainder='passthrough'
   )
   X = column_transformer.fit_transform(data.drop('bad_packet', axis=1))
   y = data['bad_packet']
   ```

### 5. **Model Training**

   Three machine learning models are trained and evaluated:
   - **Logistic Regression**
   - **Random Forest Classifier**

   Example of training the Random Forest Classifier:

   ```python
   rf_model = RandomForestClassifier(random_state=42)
   rf_model.fit(X_train, y_train)
   ```

### 6. **Model Evaluation**

   The models are evaluated using classification metrics, focusing on accuracy, precision, recall, and F1 score.

   ```python
   y_pred = rf_model.predict(X_test)
   print(classification_report(y_test, y_pred))
   ```

### 7. **Visualization**

   - Visualizations such as confusion matrices and ROC curves are included to better understand model performance.

---

## Getting Started

1. **Clone the Repository** and open the notebook in Jupyter:
   ```bash
   git clone https://github.com/SparshLadani/Network-Intrusion-Detection-System.git
   cd Network-Intrusion-Detection-System
   jupyter notebook
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

3. **Run the Notebook**:
   - Open `Network_Intrusion_Detection_System.ipynb`.
   - Execute each cell sequentially to load data, train models, and evaluate the results.

---

## Results

After running the models, the Random Forest Classifier and Logistic Regression achieved a 100% accuracy on the dataset, making it the preferred choice for deployment in a real-time intrusion detection system. Detailed classification metrics are available in the notebook.

---

## Conclusion

This Network Intrusion Detection System demonstrates the use of machine learning techniques for identifying malicious network traffic. It provides a foundation for further enhancement with more complex models, real-time data processing, and integration into network monitoring setups.

---

## Acknowledgments

- Dataset by Advait Menon, available on [Kaggle](https://www.kaggle.com/datasets/advaitnmenon/network-traffic-data-malicious-activity-detection).

--- 
