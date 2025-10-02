import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pd.set_option('display.max_columns', None)   #bütün sütunları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x)  #virgülden sonra 3 vasamak göster
pd.set_option('display.width', 500)  #konsolda gösterimi geniş tut

df = pd.read_csv("diabetes.csv")
df.head()

cols = [col for col in df.columns if "Outcome" not in col]  #bağımlı değişken dışındakileri aldık


############################################################################################################
# Data Preprocessing (Veri Ön İşleme)
############################################################################################################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):    #aykırı değer var mı yok mu?
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):   #aykırı değerleri hesaplanan eşikler ile değiştir
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")


for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

y = df["Outcome"]    #bağımlı değişkem
X = df.drop(["Outcome"], axis=1)
log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,  #modeli ver
                            X, y,       #bağımsız değişkenleri ver, modeli kullanarak bağımsız değişkenleri parçalara böldükten sonra deneme işlemleri yapacak
                            cv=5,       #5 katlı crossvalidation yapacağız
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"], return_train_score=True)

cv_results['test_accuracy'].mean()    # ortalama accuracy
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

# Eğitim ve test doğruluklarını hesapla
train_f1_mean = cv_results["train_f1"].mean()
test_f1_mean = cv_results["test_f1"].mean()
if train_f1_mean > test_f1_mean + 0.05:
    print("Model overfitting yapıyor olabilir.")
elif test_f1_mean > train_f1_mean + 0.05:
    print("Model underfitting yapıyor olabilir.")
else:
    print("Modelin performansı dengeli görünüyor.")



random_user = X.sample(1, random_state=45)   #rastgele bir kullanıcı seçiyorum
log_model.predict(random_user)