# uber_fare_predictions-ml-project
The Uber Fare Prediction project predicts Uber ride fares using machine learning models like Linear Regression and Random Forest Regressor. It integrates geo-map functionality to help users interactively select pickup and dropoff locations for real-time fare predictions. Built with Python, scikit-learn, pandas, NumPy, and map APIs, this project demonstrates the complete pipeline of data preprocessing, model development, and deployment. The data is cleaned and processed by handling missing values, correcting outliers, and calculating ride distances using the Haversine formula. A Streamlit-based web app provides a user-friendly interface, where users can easily input locations and predict ride fares. The project showcases the superiority of Random Forest Regressor over Linear Regression, with performance metrics such as R², MSE, and RMSE highlighting its accuracy.
To run the project locally, follow these steps:
Clone the repository to your local machine:git clone https://github.com/your-repo-name.git
cd uber-fare-prediction 
Launch the Streamlit app using:streamlit run app.py
Once the app is running, open your browser and navigate to http://localhost:8501 to interact with the app and start predicting Uber fares.
This project provides a powerful, interactive way to explore machine learning models and fare predictions, demonstrating how data-driven solutions can optimize pricing strategies in the transportation industry.
