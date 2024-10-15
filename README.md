# Loan Approval Prediction Model

This repository contains code and data for building machine learning models to predict loan approval status based on customer demographics and financial details. Two approaches are implemented:
- A **Self Organizing Map (SOM)** to cluster customer profiles.
- A **Classification Model** to predict whether a loan will be approved or denied.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Feature Importance](#feature-importance)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Loan approval is a critical decision for financial institutions, and accurate predictions can reduce financial risks. This project explores machine learning techniques to classify loan applications as "approved" or "not approved" based on historical customer data.

## Dataset
The dataset used for this project contains customer information such as age, income, employment status, credit history, loan amount, and others. Each record is labeled with a loan approval status:
- **Approved**: 1
- **Not Approved**: 0

The features include:
- Gender
- Marital Status
- Employment Status
- Credit History
- Income
- Age, etc.

The target variable is `Class` (0 or 1), indicating loan approval status.

## Models Implemented

### 1. **Self Organizing Map (SOM)**
- **Purpose**: Cluster customer profiles to detect groups with similar loan approval likelihood.
- **Approach**: SOM groups customers into clusters based on their features, helping to identify patterns in approved and denied loans.

### 2. **Classification Model**
- **Purpose**: Predict loan approval status based on customer data.
- **Approach**: A supervised learning model is used to classify loans as approved or not based on features like income, credit history, and employment status. We have implemented models like **Random Forest**.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Preprocess Data**: Ensure your dataset is cleaned and preprocessed (as shown in the notebooks).
   
2. **Train the Model**:
   - To train and evaluate the **SOM** model, run the `Loan_Approval.ipynb`.
   - To train the **Classification Model**, run the `Loan_Approval_using_Classification.ipynb`.

3. **Predict Loan Approval**:
   Use the trained classification model to predict the loan approval status of new customer data by feeding it through the model.

```python
# Example of using the trained classification model
predictions = classifier.predict(new_customer_data)
```

## Feature Importance
We use **Random Forest** to identify the top three features influencing loan approval decisions. These are determined based on the feature importance scores output by the model.

To view the top features, run the following code:
```python
print(feature_importance_df.head(3))
```

## Results

- **SOM Model**: Visualizes clusters of customers to identify groups with similar approval likelihood.
- **Classification Model**: Achieves accuracy of [add your result] in predicting loan approval status, with top features influencing the decision being [list top 3 features].

## Future Work
- **Hyperparameter Tuning**: Optimize models for better accuracy.
- **Feature Engineering**: Add more relevant features to improve model performance.
- **Model Deployment**: Deploy the classification model as a web API for real-time loan approval prediction.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to improve the code, add new models, or optimize existing ones.
