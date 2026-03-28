import logging
from src.exception import CustomException
from src.logger import logging
import os
from datetime import datetime
LOG_FILE="logs/log_" + datetime.now().strftime("%Y-%m-%d") + ".log"
log_path = os.path.join(os.getcwd(), LOG_FILE)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
LOG_FILE_PATH = os.path.join(os.getcwd(), LOG_FILE)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")