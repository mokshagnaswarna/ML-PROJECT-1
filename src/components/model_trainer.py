import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        print("MODEL TRAINING STARTED")  # debug

        try:
            logging.info("Splitting training and testing input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

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

            model_report = {}
            trained_models = {}

            # Train all models
            for name, model in models.items():
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

                model_report[name] = score
                trained_models[name] = model

                print(f"{name} R2 Score: {score}")

            # Best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            print(f"Best Model: {best_model_name} Score: {best_model_score}")

            # Create artifacts folder
            os.makedirs("artifacts", exist_ok=True)

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print("Model saved ✅")

            return best_model_name, best_model_score

        except Exception as e:
            print("ERROR:", e)
            raise CustomException(e, sys)
        
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)