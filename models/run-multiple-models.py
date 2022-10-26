import datetime
import numpy as np
import pandas as pd
import glob
import timeit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macosx')
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Simple time now function
def tn():
    return datetime.datetime.now().strftime("%H:%M:%S") + ": "

def getModels():
    models = [
        # ('LogisticRegression', LogisticRegression(max_iter=100000)),
        # ('RandomForestClassifier', RandomForestClassifier()),
        # ('KNeighborsClassifier', KNeighborsClassifier()),
        # ('SVM', SVC()),
        # ('GaussianNB', GaussianNB()),
        ('XGB', XGBClassifier()),
        ('XGB2', XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)),
        ('XGB3', XGBClassifier(objective='multi:softprob', enable_categorical=False,
                               eval_metric='merror', max_depth=12, n_estimators=400))
    ]
    return models


def getModelAndParams():
    #  https://xgboost.readthedocs.io/en/stable/parameter.html
    XGBParams = {
            'objective': 'multi:softprob',
            'enable_categorical': False,
            'eval_metric': 'merror',  # mlogloss or merror,
            # 'eval_metric': 'multi:softprob'
    }
    return [ XGBParams ], [ XGBClassifier ]


def plot_confusion_matrix3(cm, labels, title, wait_for_graph):
    print(cm)
    if wait_for_graph:
        cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
        cmd.plot()
        plt.title(title)
        plt.show(block=True)
        # plt.interactive(False)


def loadData(param):
    #
    # load TRADES
    print(tn() + "Starting main()")
    csv_files = glob.glob(param.p_in_directory + "*.csv")
    if len(csv_files) == 0:
        print("No files in directory ", param.p_in_directory)
        exit(1)
    print("          loading " + param.p_in_directory + " file count: " + str(len(csv_files)))
    # Read each CSV file into DataFrame
    # This creates a list of dataframes
    df_list = (pd.read_csv(file) for file in csv_files)
    # Concatenate all DataFrames
    data = pd.concat(df_list, ignore_index=True)
    print(tn() + "Loaded TRADES.", data.shape)

    data.set_index('date')
    data.sort_index()

    # Time between 945 and 1545
    data = data[pd.to_datetime(data['date']).dt.time > datetime.time(9, 45)]
    data = data[pd.to_datetime(data['date']).dt.time < datetime.time(15, 45)]
    data.dropna(inplace=True)
    print("filtering 945-1545, & na rows", data.shape)
    return data

def main(param, wait_for_graph):

    data = loadData(param)
    data.dropna()

    X_cols  = [
                'h0s_low', 'h0s_average', 'h0s_high', 'h0s_volume', 'h0s_barCount',
                'h1s_low', 'h1s_average', 'h1s_high', 'h1s_volume', 'h1s_barCount',
                'h2s_low', 'h2s_average', 'h2s_high', 'h2s_volume', 'h2s_barCount',
                'h3s_low', 'h3s_average', 'h3s_high', 'h3s_volume', 'h3s_barCount',
                'h4s_low', 'h4s_average', 'h4s_high', 'h4s_volume', 'h4s_barCount',
                #
                'h0s_ask_max', 	'h0s_ask_avg', 'h0s_bid_min', 	'h0s_bid_avg',
                'h1s_ask_max', 	'h1s_bid_min', 	'h1s_bid_avg', 	'h1s_ask_avg',
                'h2s_ask_max', 	'h2s_bid_min', 	'h2s_bid_avg', 	'h2s_ask_avg',
                'h3s_ask_max', 	'h3s_bid_min', 	'h3s_bid_avg', 	'h3s_ask_avg',
                'h4s_ask_max', 	'h4s_bid_min', 	'h4s_bid_avg', 	'h4s_ask_avg',
                #
                'vix'
               ]
    y_cols = ['f5s_average', 'f5s_10c_arrow', 'f5s_15c_arrow', 'f5s_20c_arrow', 'f5s_25c_arrow']

    X = data[X_cols]
    y = data [y_cols[1:]]
    X_train, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape, y_train_all.shape, y_test.shape)

    resAll = pd.DataFrame()
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    for y_index in range(len(y.columns)):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_all.iloc[:, y_index])
        for name, model in getModels():
            # lrn = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(model)
            title = name + ": " + y_cols[y_index] + "(" + str(y_index) + ")"
            print(title)
            print(classification_report(y_test.iloc[:, y_index], le.inverse_transform(y_pred), zero_division=0))
            cm_f10 = confusion_matrix(y_true=y_test.iloc[:,y_index], y_pred=le.inverse_transform(y_pred),
                                      labels=y_test.iloc[:,y_index].unique())
            plot_confusion_matrix3(cm_f10, labels=y_test.iloc[:, y_index].unique(),
                                   title=title, wait_for_graph=wait_for_graph)

    #
    print(resAll.columns)
    print(resAll)


class Param:
    def __init__(self, p_in_directory, p_out_directory, p_symbol, p_month_no=0):
        self.p_in_directory = p_in_directory
        self.p_out_directory = p_out_directory
        self.p_symbol = p_symbol
        self.p_month_no = p_month_no

    def __str__(self):
        return "[" + self.p_in_directory + ", " + self.p_out_directory \
               + ", " + self.p_symbol + ", " + str(self.p_month_no) + "]"


params = [
    Param("../data/projected2/", "../data/projected/models/", "TSLA", 4)
]


start_time = datetime.datetime.now()
for pm in params:
    print(tn() + " Starting Execution for ", pm)
    main(pm, wait_for_graph=False)
print("\n\nStarted: ", start_time, ' Finished: ', datetime.datetime.now(), ' Dur: ', (datetime.datetime.now() - start_time).total_seconds())