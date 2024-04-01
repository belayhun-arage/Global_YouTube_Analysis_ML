# Inside main.py

from src.data_processing import DataProcessor
from config import data_path

def main():
    # Instantiate DataProcessor with data_path argument
    processor = DataProcessor(data_path)

    # Load the dataset
    df = processor.load_dataset()
    print(df)

    # Investigate the data
    # processor.investigate_data()

if __name__ == "__main__":
    main()
