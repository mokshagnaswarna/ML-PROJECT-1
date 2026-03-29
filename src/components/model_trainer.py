import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ✅ Model save config
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# ✅ Model Trainer class
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        print("MODEL TRAINING STARTED")  # debug

        try:
            logging.info("Splitting training and testing data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ✅ Models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "SVR": SVR()
            }

            # ✅ Hyperparameters
            params = {
                "Random Forest": {"n_estimators": 100},
                "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1},
                "AdaBoost": {"n_estimators": 100},
                "Linear Regression": {},
                "KNN": {"n_neighbors": 5},
                "Decision Tree": {"criterion": "squared_error"},
                "XGBoost": {"n_estimators": 100, "learning_rate": 0.1},
                "CatBoost": {"learning_rate": 0.1, "depth": 6, "verbose": False},
                "SVR": {"kernel": "rbf"}
            }

            model_report = {}
            trained_models = {}

            # 🔁 Train all models
            for name, model in models.items():

                # apply parameters
                if name in params:
                    model.set_params(**params[name])

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

                model_report[name] = score
                trained_models[name] = model

                print(f"{name} R2 Score: {score}")

            # ✅ Best model selection
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            print(f"\nBest Model: {best_model_name} with score: {best_model_score}")

            logging.info(f"Best Model: {best_model_name}, Score: {best_model_score}")

            # ✅ Create folder
            os.makedirs("artifacts", exist_ok=True)

            # ✅ Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print("Model saved successfully ✅")

            return best_model_name, best_model_score

        except Exception as e:
            print("ERROR:", e)
            raise CustomException(e, sys)


# ✅ RUN FULL PIPELINE
if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")

    # Step 1: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)

    # Step 2: Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)