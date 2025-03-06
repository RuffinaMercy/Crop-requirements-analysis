import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the datasets
data_1 = pd.read_csv("Fertilizer Prediction.csv")  # Training dataset
data_2 = pd.read_excel("crop_requirements.xlsx")   # Crop NPK requirements dataset

# for consistency
data_1['Crop Type'] = data_1['Crop Type'].str.strip().str.lower()
data_2['Crop Type'] = data_2['Crop Type'].str.strip().str.lower()

# Merge datasets based on 'Crop Type'
merged_data = pd.merge(data_1, data_2, on="Crop Type", how="left")

# Load the nutrient content of fertilizers (DAP, Urea, MOP)
fertilizer_contents = {
    'DAP': {'N': 0.18, 'P': 0.46, 'K': 0},
    'Urea': {'N': 0.46, 'P': 0, 'K': 0},
    'MOP': {'N': 0, 'P': 0, 'K': 0.60}
}

# Fertilizer efficiency (NPK)
fertilizer_efficiency = {'N': 50, 'P': 30, 'K': 60}

# Ensure all necessary columns exist in merged_data
required_columns = ['N_crop', 'P_crop', 'K_crop', 'Nitrogen', 'Phosphorous', 'Potassium']
for col in required_columns:
    if col not in merged_data.columns:
        raise ValueError(f"Missing required column in merged data: {col}")

# Calculate DAP required (phosphorus and nitrogen from DAP)
merged_data['DAP'] = (
    ((merged_data['P_crop'] - merged_data['Phosphorous']) / (fertilizer_efficiency['P'] / 100) / fertilizer_contents['DAP']['P'])
    +
    ((merged_data['N_crop'] - merged_data['Nitrogen'] - ((merged_data['P_crop'] - merged_data['Phosphorous']) / (fertilizer_efficiency['P'] / 100) / fertilizer_contents['DAP']['P'] * fertilizer_contents['DAP']['N'] / 100)) / (fertilizer_efficiency['N'] / 100) / fertilizer_contents['DAP']['N'])
).fillna(0)

# Calculate Urea required (after excluding nitrogen from DAP)
merged_data['Urea'] = ((merged_data['N_crop'] - merged_data['Nitrogen'] - (merged_data['DAP'] * fertilizer_contents['DAP']['N'] / 100)) / (fertilizer_efficiency['N'] / 100) / fertilizer_contents['Urea']['N']).fillna(0)

# Calculate MOP required (for potassium)
merged_data['MOP'] = ((merged_data['K_crop'] - merged_data['Potassium']) / (fertilizer_efficiency['K'] / 100) / fertilizer_contents['MOP']['K']).fillna(0)

# Save the updated dataset to an Excel file
merged_data.to_excel("calc.xlsx", index=False)

print("Fertilizer requirements have been calculated and saved as 'calc.xlsx'.")

# Load the dataset (use merged_data or your dataset)
data = pd.read_excel("calc.xlsx")  # You can adjust this line if you're using another dataset

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Check the columns of the dataset
print("\nDataset columns:", data.columns)

print("\nNull value count : ")
print(data.isnull().sum())
data.dropna()
data.drop_duplicates()

# Assign unique IDs to 'Soil Type' and 'Crop Type'
data['soil type id'] = pd.factorize(data['soil type'])[0]
data['crop type id'] = pd.factorize(data['crop type'])[0]
# Features (X) and target (y)
X = data[['temparature', 'humidity', 'moisture', 'soil type id']]  # Input features
y = data['crop type id']  # Target variable



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Decision Tree Classifier
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
# Train the model
tree_model.fit(X_train, y_train)
# Predict crop types
y_pred = tree_model.predict(X_test)
# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")

# Plot the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Crop Type')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Crop Type')
plt.xlabel('Samples')
plt.ylabel('Crop Type ID')
plt.title('Decision Tree Model - Crop Type Prediction')
plt.legend()
plt.show()

def predict_crop_type_decision_tree(temperature, humidity, moisture, soil_type):
    # Ensure soil_type exists in the dataset
    if soil_type.lower() not in available_soil_types:
        raise ValueError(f"Invalid soil type: {soil_type}. Please enter a valid soil type from the list.")

    # Get the soil type ID
    soil_type_id = pd.factorize(data['soil type'])[0][data['soil type'].str.lower() == soil_type.lower()][0]

    # Create input data as a DataFrame with valid feature names
    input_data = pd.DataFrame(
        [[temperature, humidity, moisture, soil_type_id]],
        columns=['temparature', 'humidity', 'moisture', 'soil type id']
    )

    # Predict the crop type ID
    predicted_crop_id = tree_model.predict(input_data)[0]

    # Map back to crop type name
    crop_type = pd.factorize(data['crop type'])[1][int(predicted_crop_id)]

    return crop_type

# Show the available soil types to the user
available_soil_types = data['soil type'].str.lower().unique()  # Convert available soil types to lowercase

# User input
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
moisture = float(input("Enter Moisture: "))

# Show available soil types
print(f"\nAvailable soil types: {', '.join(available_soil_types)}")
soil_type = input("Enter Soil Type: ").strip().lower()

# Predict and display the crop type
predicted_crop = predict_crop_type_decision_tree(temperature, humidity, moisture, soil_type)
print(f"\nThe predicted crop type that can be grown is: {predicted_crop}")

# Ask if the user wants to calculate fertilizer requirements
calculate_fertilizer = input("\nWould you like to calculate the NPK fertilizer requirements for this crop? (yes/no): ").strip().lower()

if calculate_fertilizer == 'yes':
    # Ask the user to input NPK values for the fertilizer calculation
    n_value = float(input("\nEnter the soil nitrogen (N) value for the crop: "))
    p_value = float(input("Enter the soil phosphorus (P) value for the crop: "))
    k_value = float(input("Enter the soil potassium (K) value for the crop: "))

    # Find the NPK requirement for the crop from the dataset
    crop_data = merged_data[merged_data['Crop Type'] == predicted_crop].iloc[0]

    # Calculate the fertilizer requirements
    dap_required = (
        ((p_value - crop_data['Phosphorous']) / (fertilizer_efficiency['P'] / 100) / fertilizer_contents['DAP']['P'])
        +
        ((n_value - crop_data['Nitrogen'] - ((p_value - crop_data['Phosphorous']) / (fertilizer_efficiency['P'] / 100) / fertilizer_contents['DAP']['P'] * fertilizer_contents['DAP']['N'] / 100)) / (fertilizer_efficiency['N'] / 100) / fertilizer_contents['DAP']['N']))

    urea_required = ((n_value - crop_data['Nitrogen'] - (dap_required * fertilizer_contents['DAP']['N'] / 100)) / (fertilizer_efficiency['N'] / 100) / fertilizer_contents['Urea']['N'])

    mop_required = ((k_value - crop_data['Potassium']) / (fertilizer_efficiency['K'] / 100) / fertilizer_contents['MOP']['K'])

    # Print the fertilizer requirements
    print(f"\nFor crop type '{predicted_crop}', the required fertilizers are:")
    print(f"DAP: {dap_required:.2f} kg")
    print(f"Urea: {urea_required:.2f} kg")
    print(f"MOP: {mop_required:.2f} kg")
else:
    print("\nExiting step. ")

from sklearn.tree import plot_tree

decisionTree = input("\nVisulaize your decision tree ?(yes/no)").strip().lower()
# Visualize the Decision Tree
if decisionTree == 'yes':
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model,
              feature_names=X.columns,
              class_names=pd.factorize(data['crop type'])[1],
              filled=True,
              rounded=True)
    plt.title("Decision Tree Visualization")
    plt.show()
else :
    print("\nExiting the program. ")
