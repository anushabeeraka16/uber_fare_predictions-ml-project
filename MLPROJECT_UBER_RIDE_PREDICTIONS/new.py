import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math

# Haversine formula to calculate the distance between two points on the Earth
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)*2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)*2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Radius of the Earth in kilometers
    radius = 6371.0
    
    # Calculate the distance
    distance = radius * c
    
    return distance

# App title with cute icon
st.title("🚖 Uber Fare Prediction App")
st.markdown("An interactive app to analyze and predict Uber ride fares. 🛻")

# Sidebar menu
st.sidebar.title("🔍 Navigation")
options = st.sidebar.radio(
    "Select a feature:",
    ["Dataset Overview and Preprocessing", "Data Visualization", "Model Selection", "Evaluation Metrics", "Fare Prediction"]
)

# Initialize session state for the dataset
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# Dataset Overview and Preprocessing
if options == "Dataset Overview and Preprocessing":
    st.subheader("📊 Dataset Overview and Preprocessing")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file:
        # Load the dataset and store it in session state
        data = pd.read_csv(uploaded_file)
        st.session_state.dataset = data
        st.success("Dataset uploaded successfully!")

    # Display the dataset if already uploaded
    if st.session_state.dataset is not None:
        st.write("Preview of the dataset:")
        st.dataframe(st.session_state.dataset.head())

        # Handle unnamed columns
        st.write("Removing 'Unnamed' columns if any:")
        st.session_state.dataset = st.session_state.dataset.loc[:, ~st.session_state.dataset.columns.str.contains('^Unnamed')]

        # Display dataset after dropping unnamed columns
        st.write("Dataset after removing 'Unnamed' columns:")
        st.dataframe(st.session_state.dataset.head())

        # Summary statistics
        st.write("Summary statistics:")
        st.write(st.session_state.dataset.describe())

        # Data Preprocessing
        # Filling missing values with mean
        st.write("Handling missing values:")
        if st.checkbox("Fill missing values with mean for numerical columns"):
            for col in st.session_state.dataset.select_dtypes(include=[np.number]).columns:
                st.session_state.dataset[col].fillna(st.session_state.dataset[col].mean(), inplace=True)
            st.success("Missing values in numerical columns filled with mean.")

        # Convert pickup_datetime from object to datetime format
        st.write("Converting 'pickup_datetime' to datetime format:")
        st.session_state.dataset['pickup_datetime'] = pd.to_datetime(st.session_state.dataset['pickup_datetime'], errors='coerce')
        st.success("'pickup_datetime' converted to datetime format.")

        # Dropping 'pickup_datetime' column
        if st.checkbox("Drop 'pickup_datetime' column"):
            st.session_state.dataset.drop(columns=['pickup_datetime'], inplace=True)
            st.success("'pickup_datetime' column dropped.")

        # Handling outliers using the clip function
        st.write("Handling outliers:")
        for col in st.session_state.dataset.select_dtypes(include=[np.number]).columns:
            q1 = st.session_state.dataset[col].quantile(0.25)
            q3 = st.session_state.dataset[col].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            st.session_state.dataset[col] = st.session_state.dataset[col].clip(lower=lower_limit, upper=upper_limit)
        st.success("Outliers handled using the clip function.")

        # Handling incorrect coordinates (distance should be below 130 km)
        st.write("Handling incorrect coordinates:")
        st.session_state.dataset = st.session_state.dataset[
            (st.session_state.dataset['pickup_latitude'].between(-90, 90)) &
            (st.session_state.dataset['pickup_longitude'].between(-180, 180)) &
            (st.session_state.dataset['dropoff_latitude'].between(-90, 90)) &
            (st.session_state.dataset['dropoff_longitude'].between(-180, 180))
        ]
        st.success("Incorrect coordinates filtered.")

# Data Visualization
elif options == "Data Visualization":
    st.subheader("📈 Data Visualization")

    if st.session_state.dataset is not None:
        data = st.session_state.dataset

        # Show Boxplots for every feature when checkbox is clicked
        if st.checkbox("Show Boxplot for every feature"):
            st.write("Boxplot for numerical features:")
            num_cols = data.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=data[col], ax=ax)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)

        # Correlation Heatmap after handling outliers
        if st.checkbox("Show Correlation Heatmap"):
            st.write("Correlation heatmap:")

            # Ensure that the dataset has only numerical values for correlation
            numeric_data = data.select_dtypes(include=[np.number])

            # Handle missing values (fill with mean for example)
            numeric_data = numeric_data.fillna(numeric_data.mean())

            # Calculate the correlation matrix
            corr = numeric_data.corr()

            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
         # Show Histograms when checkbox is clicked
        if st.checkbox("Show Histograms for numerical features"):
            st.write("Histograms for numerical features:")
            num_cols = data.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data[col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

# Model Selection
elif options == "Model Selection":
    st.subheader("🤖 Train and Compare Models")

    if st.session_state.dataset is not None:
        data = st.session_state.dataset
        # Using Haversine for calculating the distance
        def haversine(lat1, lon1, lat2, lon2):
            p = 0.017453292519943295  # PI/180
            a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
            return 12742 * np.arcsin(np.sqrt(a))  # 2 * R = 12742 km

        # Calculate distance using Haversine
        data['distance'] = haversine(data['pickup_latitude'], data['pickup_longitude'], data['dropoff_latitude'], data['dropoff_longitude'])

        # Remove unrealistic distances (above 130 km)
        data = data[data['distance'] <= 130]

        X = data[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'distance']]
        y = data['fare_amount']

        # Split data (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model options
        model_choice = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest Regressor"])

        if st.button("Train Model"):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest Regressor":
                model = RandomForestRegressor(n_estimators=10, random_state=42)

            # Train the model
            model.fit(X_train, y_train)

            # Save model for later
            with open("trained_model1.sav", "wb") as file:
                pickle.dump(model, file)

            predictions = model.predict(X_test)
            metrics = {
                "R2 Score": r2_score(y_test, predictions),
                "MSE": mean_squared_error(y_test, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
            }

            # Save evaluation metrics
            with open("evaluation_metrics.pkl", "wb") as metrics_file:
                pickle.dump(metrics, metrics_file)

            st.success(f"{model_choice} trained successfully!")
            st.write("Navigate to 'Evaluation Metrics' to see the model performance.")

# Evaluation Metrics
elif options == "Evaluation Metrics":
    st.subheader("📊 Model Evaluation Metrics")

    try:
        with open("evaluation_metrics.pkl", "rb") as metrics_file:
            metrics = pickle.load(metrics_file)

        st.write("Model Performance:")
        st.write(f"*R² Score*: {metrics['R2 Score']:.2f}")
        st.write(f"*Mean Squared Error (MSE)*: {metrics['MSE']:.2f}")
        st.write(f"*Root Mean Squared Error (RMSE)*: {metrics['RMSE']:.2f}")
    except FileNotFoundError:
        st.error("No evaluation metrics found. Train a model in the 'Model Selection' section first.")

# Fare Prediction
elif options == "Fare Prediction":
    st.subheader("💵 Predict Fare")

    try:
        with open("trained_model1.sav", "rb") as file:
            model = pickle.load(file)

        st.success("Model loaded successfully!")

        # Move input fields from sidebar to main region
        st.header("Enter Ride Details")

        pickup_lat = st.number_input("Pickup Latitude", format="%.6f")
        pickup_lon = st.number_input("Pickup Longitude", format="%.6f")
        dropoff_lat = st.number_input("Dropoff Latitude", format="%.6f")
        dropoff_lon = st.number_input("Dropoff Longitude", format="%.6f")
        
        # Calculate distance
        distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

        # Move the button to the main region
        if st.button("Predict Fare"):
            input_data = pd.DataFrame([[pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, distance]],
                                      columns=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'distance'])
            predicted_fare = model.predict(input_data)
            st.write(f"Predicted fare: ${predicted_fare[0]:.2f}")

    except FileNotFoundError:
        st.error("No trained model found. Train a model in the 'Model Selection' section first.")
