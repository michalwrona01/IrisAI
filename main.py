from sklearn import preprocessing
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)

names_iris = {'Iris-setosa': 0,
              'Iris-versicolor': 1,
              'Iris-virginica': 2}


def get_row_with_normalize_name(row, names_iris):
    row[-1] = names_iris[row[-1]]
    return row


if __name__ == "__main__":
    csv_file_pd = pd.read_csv('data/iris.csv')

    csv_data_list = csv_file_pd.values.tolist()
    csv_data_with_normalized_names_list = [get_row_with_normalize_name(row, names_iris)
                                           for row in csv_data_list
                                           if row]

    columns_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'name']

    csv_data_with_normalized_names_dataframe = pd.DataFrame(csv_data_with_normalized_names_list,
                                                            columns=columns_names)

    min_max_Scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
    col = csv_data_with_normalized_names_dataframe.columns
    result = min_max_Scalar.fit_transform(csv_data_with_normalized_names_dataframe)
    min_max_Scalar_df = pd.DataFrame(result, columns=col)

    print(min_max_Scalar_df)
