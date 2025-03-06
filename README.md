

# Fertilizer Prediction and Crop Type Classification

This project involves predicting the fertilizer requirements (NPK values) for different crops based on soil conditions and crop type, and classifying crop types based on environmental variables such as temperature, humidity, and moisture. The project uses machine learning techniques with a Decision Tree Classifier to predict the crop type and calculate the required fertilizers (DAP, Urea, and MOP) based on NPK requirements.

## Features

- Data Merging and Preprocessing: Combines a fertilizer prediction dataset with crop NPK requirement data to calculate the fertilizer amounts (DAP, Urea, MOP) required for different crops.
- Decision Tree Model: Trains a Decision Tree Classifier model to predict crop types based on environmental features (temperature, humidity, moisture, and soil type).
- Fertilizer Requirement Calculation: Based on predicted crop type, the user can calculate the amount of NPK fertilizers (DAP, Urea, MOP) required to meet crop nutrient needs.
- Visualization: Visualizes the Decision Tree model to help understand how the model makes its predictions.
- User Input for Predictions: Users can input temperature, humidity, moisture, and soil type to predict the crop type, and optionally calculate the required fertilizers for the predicted crop.

## Files Used

1.Fertilizer Prediction.csv - Dataset containing information about fertilizer requirements for different crops.
2.crop_requirements.xlsx - Dataset containing the NPK requirements for different crops.
3.calc.xlsx - Output file with calculated fertilizer requirements for crops.

## Steps

1. Preprocessing: The datasets are cleaned and merged based on crop type. Fertilizer requirements (DAP, Urea, MOP) are calculated based on the crop's nitrogen, phosphorus, and potassium needs.
2. Model Training: A Decision Tree Classifier model is trained on environmental features (temperature, humidity, moisture, soil type) to predict the crop type.
3. User Interaction: The user can input environmental conditions (temperature, humidity, moisture, soil type) and the model will predict the crop type. The user can then choose to calculate the required fertilizers (DAP, Urea, MOP) for the predicted crop.
4. Model Visualization: Users can visualize the Decision Tree to understand the model's decision-making process.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python script to start the application.

## Usage

1. Input environmental data (temperature, humidity, moisture, soil type) when prompted.
2. View the predicted crop type.
3. Optionally, calculate the required fertilizers (DAP, Urea, MOP) based on the predicted crop type.

## Example Usage


Enter Temperature: 30.5

Enter Humidity: 65

Enter Moisture: 20

Available soil types: sandy, loamy, clayey

Enter Soil Type: loamy

The predicted crop type that can be grown is: Wheat

Would you like to calculate the NPK fertilizer requirements for this crop? (yes/no): yes

Enter the soil nitrogen (N) value for the crop: 15

Enter the soil phosphorus (P) value for the crop: 10

Enter the soil potassium (K) value for the crop: 5

For crop type 'Wheat', the required fertilizers are:

DAP: 25.34 kg

Urea: 16.21 kg

MOP: 10.56 kg
```

