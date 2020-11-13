import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MarketingAnalysis:

    def __init__(self):
        self.data = pd.read_csv('Data/term-deposit-marketing-2020.csv', sep=",")


    def transform(self,data):

        all_labelencoders = {}
        column_name=data.columns

        for i in  column_name:
            print(i)

            labelencoder = LabelEncoder()
            if data[i].dtype == object:
                all_labelencoders[i] = labelencoder
                labelencoder.fit(data[i])
                data['labelencoder_' + i] = labelencoder.transform(data[i])
                data = data.drop([i], axis=1)
        print(data.head(5))
        corr_matrix=data.corr()
        print("corr_matrix",corr_matrix)
        sn.heatmap(corr_matrix, annot=True)
        print(plt.show())

        return data

    def smote(self, data):
        k = 5
        seed = 100
        column_name=data.columns
        X = data.iloc[:, :-1].values
        y= data.iloc[:,-1].values
        sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
        X_res, y_res = sm.fit_resample(X, y)
        data = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
        data.columns=column_name

        data = data.drop(columns=['duration'])
        data = data.drop(columns=['labelencoder_contact'])
        data = data.drop(columns=['labelencoder_housing'])


        data = shuffle(data)
        return data
        # rename the columns

    def model(self,data):

        X = data.iloc[:, :-1].values
        y= data.iloc[:,-1].values
        model = xgb.XGBClassifier()
        kfold = KFold(n_splits=10, random_state=7 ,shuffle=True)
        results = cross_val_score(model, X, y, cv=kfold)

        return results

    def model2(self,data):
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
        # fit model on all training data
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        for col, score in zip(X_train.columns, model.feature_importances_):
            print("feature_importance",col, score)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def result(self):
        result_data = self.transform(self.data)
        result_data = self.smote(result_data)
        result_accuracy2 = self.model2(result_data)
        result_accuracy = self.model(result_data)
        print("Accuracy: %.2f%% (%.2f%%)" % (result_accuracy.mean() * 100, result_accuracy.std() * 100))
        print("Accuracy2: %.2f%%" % (result_accuracy2 * 100.0))

result = MarketingAnalysis()
result_accuracy=result.result()
