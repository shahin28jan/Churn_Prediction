import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")

    ## Create a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
            
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")
        try:
            df=pd.read_excel(os.path.join('customer_churn_large_dataset.xlsx'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            #Rename all columns

            logging.info('Train test split')

            df = df.drop(labels=['CustomerID', 'Name'],axis=1)
            
            train_set,test_set=train_test_split(df,test_size=0.30)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')


            return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')   
            raise CustomException(e,sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    #data_transformation.initaite_data_transformation()

