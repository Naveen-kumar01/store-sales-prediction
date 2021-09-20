"""
This is the Entry point for Training the Machine Learning Model.

Written By: iNeuron Intelligence
Version: 1.0
Revisions: None

"""


# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from best_model_finder import tuner
from file_operations import file_methods
from app_logging import logger
import numpy as np
import pandas as pd

#Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_logs/ModelTrainingLog.txt", 'a+')
    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_data()


            """doing the data preprocessing"""

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            # imputing missing values in the dataset
            data = preprocessor.impute_missing_values_num(self, data, "Item_Weight", "Item_Identifier")
            data = preprocessor.impute_missing_values_cat(self, data, "Outlet_Size", "Outlet_Type")
            data = preprocessor.impute_missing_values_not_null(self, data, "Item_Visibility", "Item_Identifier")
            # extracting new features from the item type variable
            data = preprocessor.feature_extract(data, "Item_Type_Combined")
            # extracting age from the year attributes
            data['Outlet_Years'] = 2021 - data['Outlet_Establishment_Year']
            # imputing the non valuable attribute into valuable and meaningfull attribute
            data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat',
                                                                         'reg': 'Regular',
                                                                         'low fat': 'Low Fat'})
            data.loc[data['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"

            # Label encoding of categorical variable
            column = "Outlet_Identifier"
            data = preprocessor.Label_Encode(data, column)
            column_list = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type',
                       'Outlet']
            data = preprocessor.Label_Encode_col(data, column_list)

            # one hot encoding of categorical variable
            data = preprocessor.one_hot_encoding(data, column_list)

            # remove the columns that are not required for prediction
            data = preprocessor.remove_columns(data, ['Item_Type','Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier'])

            # create separate features and labels
            X,Y = preprocessor.separate_label_feature(data, label_column_name='Item_Outlet_Sales')

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if(is_null_present):
                X = preprocessor.impute_missing_values(X, cols_with_missing_values) # missing value imputation

                # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=355)

            model_finder = tuner.Model_Finder(self.file_object, self.log_writer) # object initialization

            #getting the best model for each of the clusters
            best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)

            #saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object, self.log_writer)
            save_model = file_op.save_model(best_model, best_model_name)

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise e