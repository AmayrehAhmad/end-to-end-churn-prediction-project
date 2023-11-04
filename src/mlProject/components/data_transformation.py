import os
import pickle

from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    def preprocessing(self, df):
        exclude_columns = list(self.config.exclude_columns)
        df = df.drop(columns=exclude_columns, axis=1)

        path_encoders = self.config.encoders_path
        with open(path_encoders, "rb") as file:
            dict_encoders = pickle.load(file)

        columns_to_encode = list(dict_encoders.keys())

        for col in columns_to_encode:
            df[col] = dict_encoders[col].transform(df[col])


        target_column = self.config.target_column

        if target_column in df.columns:
            dict_labels = self.config.dict_lables
            df[target_column] = df[target_column].map(dict_labels)

        return df


    def prepare_encoders(self, df):
        exclude_columns = list(self.config.exclude_columns)


        df = df.drop(columns=exclude_columns, axis=1)

        categorical_columns = self.config.categorical_columns
        categorical_columns = list(filter(lambda x: x not in exclude_columns, categorical_columns))

        binary_columns = self.config.binary_columns
        binary_columns = list(filter(lambda x: x not in exclude_columns, binary_columns))

        # use Label encoder to encode categorical and binary columns
        dict_encoders = {}
        for col in categorical_columns+binary_columns:
            encoder = LabelEncoder()
            encoder.fit(df[col])
            dict_encoders[col] = encoder


        with open(self.config.encoders_path, "wb") as file:
            pickle.dump(dict_encoders,file)



    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        self.prepare_encoders(data)

        data = self.preprocessing(data)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        