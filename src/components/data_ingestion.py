# src/components/data_ingestion.py
import os
import src
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    # CHANGE this to your actual raw CSV
    source_data_path: str = os.path.join("data", r"D:\\sentiment_analysis_project\\notebook\\data\\data.csv")
    train_data_path: str  = os.path.join("artifacts", "train.csv")
    test_data_path: str   = os.path.join("artifacts", "test.csv")
    raw_data_path: str    = os.path.join("artifacts", "raw.csv")
    test_size: float      = 0.2
    random_state: int     = 24
    stratify_col: str     = "target"   # will stratify if this column exists

class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion component")
        try:
            src = self.ingestion_config.source_data_path
            logging.info(f"Reading dataset from: {src}")
            df = pd.read_csv(src, encoding="ISO-8859-1")

            logging.info(f"Read dataframe with shape {df.shape}")

            # ensure artifacts dir exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # save raw snapshot (no cleaning here)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            if self.ingestion_config.stratify_col in df.columns:
                train_set, test_set = train_test_split(
                    df,
                    test_size=self.ingestion_config.test_size,
                    random_state=self.ingestion_config.random_state,
                    stratify=df[self.ingestion_config.stratify_col],
                    shuffle=True,
                )
            else:
                train_set, test_set = train_test_split(
                    df,
                    test_size=self.ingestion_config.test_size,
                    random_state=self.ingestion_config.random_state,
                    shuffle=True,
                )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    # train_data, test_data = obj.initiate_data_ingestion()

    # from src.components.data_transformation import DataTransformation
    # from src.components.model_trainer import ModelTrainer

    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
