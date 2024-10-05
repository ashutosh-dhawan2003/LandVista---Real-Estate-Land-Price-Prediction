# LandVista - Real Estate Land Price Prediction

LandVista is a comprehensive machine learning project aimed at predicting the prices of real estate land based on a variety of influential features. This project applies end-to-end data analysis, preprocessing, and machine learning techniques to model land prices, helping real estate investors, property developers, and analysts make informed decisions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Model Workflow](#model-workflow)
5. [Evaluation](#evaluation)
6. [How to Use the Model](#how-to-use-the-model)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)
9. [Contributing](#contributing)

## Project Overview

LandVista employs machine learning models to predict the price of real estate based on multiple property features such as proximity to infrastructure, land area, zoning classifications, and surrounding environmental factors. The primary goal is to automate and optimize the price evaluation process.

### Key Features
- **End-to-end machine learning pipeline** from data preprocessing, model selection, to evaluation.
- **Random Forest Regressor** used as the primary prediction model.
- Model evaluation using **Root Mean Squared Error (RMSE)** and feature importance analysis.
- Easy-to-use deployment for real-world predictions via saved model artifacts.

## Dataset

The dataset used for this project contains several key features known to influence real estate prices. Key columns include:

- **Land Area**: The size of the land in square meters.
- **Zoning Classifications**: The zoning type (residential, commercial, industrial, etc.).
- **Proximity to Amenities**: Distance to key amenities like schools, parks, and shopping centers.
- **Surrounding Property Prices**: Prices of nearby properties.
- **Environmental Conditions**: Air quality, noise pollution, and other environmental factors.

### Data Sources
While the exact data source is proprietary, similar datasets can be found in publicly available real estate databases, municipal zoning records, and open datasets like Kaggle real estate prices or Zillow property datasets.

### Data Processing

To make the data usable for machine learning, several preprocessing steps were applied:

- **Missing Data Handling**: Columns with missing values were imputed using median or mean values.
- **Feature Scaling**: Numerical features were standardized using Scikit-learn's `StandardScaler`.
- **Categorical Encoding**: Categorical features such as zoning classifications were encoded using one-hot encoding.
- **Train-Test Split**: Data was split into training (80%) and test (20%) sets to ensure proper evaluation.
- **Feature Engineering**: Additional features such as land price per square meter were engineered from the original dataset.

## Installation

To run this project locally, ensure you have Python 3.8+ installed. You can set up a virtual environment and install the necessary packages by following these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ashutosh-dhawan2003/LandVista.git
   cd LandVista
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
The project uses the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib (for saving/loading models)
  
You can also install them manually using:
```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

## Model Workflow

The project follows a structured machine learning workflow:

### 1. Data Preprocessing
- **Handling missing data**: Missing numerical values were filled with the median of each column.
- **Feature scaling**: Applied `StandardScaler` to scale the numerical features.
- **Encoding categorical features**: One-hot encoding was applied to categorical variables like zoning types.
- **Train-Test Split**: The dataset was split into a training set and a test set in an 80:20 ratio.

### 2. Model Training
The following machine learning models were evaluated:
- **Linear Regression**
- **Decision Trees**
- **Random Forest Regressor**

After comparing these models, the **Random Forest Regressor** was selected as the final model due to its superior performance in predicting real estate prices.

### 3. Model Evaluation
- The **Root Mean Squared Error (RMSE)** metric was used to evaluate model performance on the test data.
- The final RMSE on the test set was **2.96**, indicating a reasonably accurate prediction performance.
- **Feature Importance Analysis**: An analysis of feature importance showed that land area, proximity to schools, and zoning classifications were key factors influencing land prices.

### 4. Model Persistence
The trained Random Forest model is saved as a `.joblib` file for easy reuse in real-world applications. It can be loaded to make predictions without retraining the model.

## Evaluation

The Random Forest Regressor was evaluated on the test data using RMSE. Here’s a breakdown of model performance:

- **Train RMSE**: ~2.1
- **Test RMSE**: ~2.96
- **Feature Importance**: Key features influencing the price prediction were land area, proximity to infrastructure, and environmental quality.

The model was able to predict real estate land prices with good accuracy, though there is room for improvement with more complex models or additional data.

## How to Use the Model

### Using Pre-Trained Model

You can use the pre-trained model provided (`LandVista.joblib`) to make predictions:

1. Load the model:
   ```python
   from joblib import load
   import numpy as np

   # Load the saved Random Forest model
   model = load('LandVista.joblib')

   # Example of standardized feature input
   features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
                         -0.23748762, -1.31238772,  2.61111401, -1.0016859, -0.5778192,
                         -0.97491834,  0.41164221, -0.86091034]])

   # Predict land price
   predicted_price = model.predict(features)
   print(predicted_price)
   ```

2. Adjust the input features to match the properties of the land you want to predict.

## Future Enhancements

Several future enhancements can be made to the project:

1. **Incorporate More Features**:
   - Include economic indicators such as local employment rates, crime statistics, and population growth.
   - Use geographical data like latitude and longitude for more accurate location-based predictions.

2. **Advanced Models**:
   - Experiment with more advanced models like XGBoost, Gradient Boosting Machines, or Neural Networks to improve prediction accuracy.

3. **Hyperparameter Tuning**:
   - Fine-tune hyperparameters using techniques like GridSearchCV or RandomizedSearchCV to further improve model performance.

4. **Deployment**:
   - Build a simple web interface or API for users to input land details and get predictions instantly.
     
## Contributing

Contributions to LandVista are welcome. If you have suggestions or improvements, feel free to submit a pull request or open an issue.
