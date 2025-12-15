import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import pandas as pd

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)


formatter = logging.Formatter('%(asctime)s -%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path):

    try:
        with open(params_path,"r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retreieved from %s", params_path)
        return params
    except FileExistsError:
        logger.error("File not Found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error %s", e)
        raise


def load_data(data_url):
    try :
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s",data_url)
        return df
    except pd.errors.ParseError as e:
        logger.error("Failed to parse the CSV file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected erroe occured: %s",e)
        raise

def preprocess_data(df):
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df=df[df['clean_comment'].str.strip() != ""]

        logger.debug("Data preprocessing completed: Missing values,duplicated, and empty strings removed")
        return df
    except KeyError as e:
        logger.error("Missing column in dataframe: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocesing: %s",e)
        raise


def save_data(train_data,test_data,data_path):
    try:
        raw_data_path = os.path.join(data_path,"raw")

        os.makedirs(raw_data_path,exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)

        logger.debug("Train and test data safed to %s",raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occured while saving the data: %s",e)
        raise






def main():
    try:
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../params.yaml"))
        test_size = params["data_ingestion"]['test_size']

        df =load_data(data_url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")

        final_df = preprocess_data(df)

        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)

        save_data(train_data=train_data,test_data=test_data,data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../data"))

    except Exception as e:
        logger.error("Failed to complete data ingestion: %s",e)
        print(f"Error: {e}")

if __name__=="__main__":
    main()


