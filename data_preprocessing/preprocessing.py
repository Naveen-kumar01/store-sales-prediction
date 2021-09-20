import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def impute_missing_values_num(self, data, col1, col2):
        """
            This method is used to impute the appropriate item weight using the item identifier col as reference
            Output : Pandas dataframe
            Exception: raise error
        """
        try:
            item_avg_weight = data.pivot_table(values=col1, index=col2)
            # Get a boolean variable specifying missing Item_Weight values
            miss_bool = data[col1].isnull()
            # Impute data and check #missing values before and after imputation to confirm
            data.loc[miss_bool, col1] = data.loc[miss_bool, col2].apply(lambda x: item_avg_weight.at[x, col1])
            return data
        except Exception as e:
            raise e

    def impute_missing_values_cat(self, data, col1, col2):
        """
            This method is used to impute the appropriate Outlet_Size using the Outlet_Type col as reference
            Output : Pandas dataframe
            Exception: raise error
        """
        try:
            outlet_size_mode = data.pivot_table(values=col1, columns=col2,aggfunc=(lambda x: mode(x).mode[0]))
            # Get a boolean variable specifying missing Item_Weight values
            miss_bool = data[col1].isnull()
            # Impute data and check #missing values before and after imputation to confirm
            data.loc[miss_bool, col1] = data.loc[miss_bool, col2].apply(lambda x: outlet_size_mode[x])
            return data
        except Exception as e:
            raise e

    def impute_missing_values_not_null(self, data, col1, col2):
        """
            This method is used to impute the appropriate Item_Visibility using the item identifier col as reference
            Output : Pandas dataframe
            Exception: raise error
        """
        try:
            visibility_avg = data.pivot_table(values=col1, index=col2)
            # Impute 0 values with mean visibility of that product:
            miss_bool = (data[col1] == 0)
            data.loc[miss_bool, col1] = data.loc[miss_bool, col2].apply(lambda x: visibility_avg[x])
            return data
        except Exception as e:
            raise e

    def feature_extract(self, data, column):
        try:
            data[column] = data['Item_Identifier'].apply(lambda x: x[0:2])
            # Rename them to more intuitive categories:
            data[column] = data[column].map({'FD': 'Food',
                                             'NC': 'Non-Consumable',
                                             'DR': 'Drinks'})
            return data
        except Exception as e:
            raise e

    def Label_Encode(self,data,column):
        try:
            le = LabelEncoder()
            data['Outlet'] = le.fit_transform(data[column])
            return data
        except:
            pass

    def Label_Encode_col(self, data, column_list):
        try:
            le = LabelEncoder()
            for i in column_list:
                data[i] = le.fit_transform(data[i])
            return data
        except Exception as e:
            raise e

    def one_hot_encoding(self, data, column_list):
        try:
            data = pd.get_dummies(data, column_list)
            return data
        except Exception as e:
            raise e


    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1)
            self.logger_object.log(self.file_object,'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels=label_column_name, axis=1) # drop the columns specified and separate the feature columns
            self.Y = data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self, data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                        returns the list of columns for which null values are present.
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            self.null_counts = data.isna().sum() # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present): # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data, cols_with_missing_values):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception
                     """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        self.cols_with_missing_values = cols_with_missing_values
        try:
            self.imputer = SimpleImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()