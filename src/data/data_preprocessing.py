import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import string
import os

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)


formatter = logging.Formatter('%(asctime)s -%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



nltk.download("wordnet")
nltk.download("stopwords")


def preprocess_comment(comment):
    try:
        comment = comment.lower()

        comment = comment.strip()

        comment = re.sub(r'\n', " ",comment)

        comment = re.sub(r'[^A-Za-z0-9\s!?.,]',"", comment)

         # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment
    
def normalize_text(df):
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug("Text normalization complete")
        return df
    except Exception as e:
        logger.error(f"Error during normaliation: {e}")
        raise

def save_data(train_data,test_data,data_path):
    try:
        interim_data_path = os.path.join(data_path,"interim")
        logger.debug(f"Creating directory {interim_data_path}")

        os.makedirs(interim_data_path,exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path,"train_processed.csv"))
        test_data.to_csv(os.path.join(interim_data_path,"test_processed.csv"))

        logger.debug(f"Processed data saved to {interim_data_path} ")

    except Exception as e:
        logger.error(f"Error occured while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing..")

        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded sucessfully")

        train_data_processed = normalize_text(train_data)
        test_data_processed = normalize_text(test_data)

        save_data(train_data_processed,test_data_processed,data_path='./data')

    except Exception as e:
        logger.error("failed to complete data preprocessng process %s",e)
        print(f"Error: {e}")

if __name__ =="__main__":
    main()







