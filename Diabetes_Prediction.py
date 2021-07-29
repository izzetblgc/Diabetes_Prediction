import numpy as np
import pandas as pd
from helpers.eda import *
from helpers.data_prep import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

dff = pd.read_csv("datasets/diabetes.csv")
df = dff.copy()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.isnull().sum() # NaN değer yok

for col in num_cols:
    num_summary(df,col,plot=False)

# Sıfır olmaması gereken değerlerin 0 olduğunu görüyoruz. Bunları NaN ile değiştirmeliyiz.

na_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

for col in na_cols:
    df[col].replace(0, np.NaN, inplace=True)

df.head()

for col in num_cols:
    num_summary(df,col,plot=False)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

################################################################
# Yeni değişkenlerin oluşturulması ve eksik değerlerin doldurulması
################################################################

#Age
df.loc[(df["Age"]>40), "Risk_Group_Age"] = "yes"
df.loc[(df["Age"]<=40), "Risk_Group_Age"] = "no"

df.loc[(df["Age"]<=30), "Age_CAT"] = "young"
df.loc[(df["Age"]>30) & (df["Age"]<=50), "Age_CAT"] = "mature"
df.loc[(df["Age"]>50), "Age_CAT"] = "senior"

# BMI'daki NaN değerleri Age_CAT değişkeninin ortalamasıyla dolduralım
df["BMI"] = df["BMI"].fillna(df.groupby("Age_CAT")["BMI"].transform("mean"))

#BMI
df.loc[(df["BMI"] <=20) , "BMI_CAT"] = "underweight"
df.loc[(df["BMI"]>20) &(df["BMI"]<=25), "BMI_CAT"] = "normal weight"
df.loc[(df["BMI"]>25) &(df["BMI"]<=30), "BMI_CAT"] = "fat"
df.loc[(df["BMI"]>30) &(df["BMI"]<=35), "BMI_CAT"] = "obese"
df.loc[(df["BMI"]>35), "BMI_CAT"] = "severely obese"

#SkinThickness NaN değerlerini BMI_CAT ortalamasıyla dolduralım
df["SkinThickness"] = df["SkinThickness"].fillna(df.groupby("BMI_CAT")["SkinThickness"].transform("mean"))

# BMI x SkinThickness
df["BMI_Skin"] = df["BMI"] * df["SkinThickness"]

#Pregnancy
df.loc[(df["Pregnancies"] < 1 ), "Have_Child"] = "no"
df.loc[(df["Pregnancies"] >= 1 ), "Have_Child"] = "yes"

# Glucose'da 5 Tane NaN değeri var. Bunları Glucose ortalamasıyla doldurabiliriz

df["Glucose"] = df["Glucose"].fillna(df["Glucose"].mean())

#Glucose
df.loc[(df["Glucose"]<=140) , "Glucose_CAT"] = "normal"
df.loc[(df["Glucose"]>140) & (df["Glucose"]<=199), "Glucose_CAT"] = "prediabetes"
df.loc[(df["Glucose"]>=200) , "Glucose_CAT"] = "diabates"

#Insulin NaN değerlerini Glucose_CAT ortalamasıyla dolduralım

df["Insulin"] = df["Insulin"].fillna(df.groupby("Glucose_CAT")["Insulin"].transform("mean"))

#BloodPressure NaN değerlerini Age_CAT ve BMI_CAT ortalamaları ile dolduralım

df["BloodPressure"] = df["BloodPressure"].fillna(df.groupby(["Age_CAT","BMI_CAT"])["BloodPressure"].transform("mean"))

df.loc[(df["BloodPressure"] <= 60 ), "BloodPressure_CAT"] = "low"
df.loc[(df["BloodPressure"] > 60) & (df["BloodPressure"] <= 90), "BloodPressure_CAT"] = "ideal"
df.loc[(df["BloodPressure"] > 90) , "BloodPressure_CAT"] = "high"

# Glucose x DiabetesPedigreeFunction
df["Glucose_Pedigree"] = df["Glucose"] * df["DiabetesPedigreeFunction"]

# Aykırı Değerler #

#Yeni değişkenlerden sonra tekrar analiz yapalım
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)


#Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype == "O"
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#One Hot Encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#Scaling

for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])

# Model
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25, random_state=26)

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_

# Tahmin

# Tahmin'lerin oluşturulması ve kaydedilmesi
y_pred_train = log_model.predict(X_train)


# Başarı Değerlendirme #

# Train Accuracy
accuracy_score(y_train, y_pred_train)

# Test
# AUC Score için y_prob
y_prob = log_model.predict_proba(X_test)[:, 1]

# Diğer metrikler için y_pred
y_pred = log_model.predict(X_test)

# CONFUSION MATRIX
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# ACCURACY
accuracy_score(y_test, y_pred)  # 0.8125

# PRECISION
precision_score(y_test, y_pred) # 0.711864406779661

# RECALL
recall_score(y_test, y_pred)  # 0.6885245901639344

# F1
f1_score(y_test, y_pred)  # 0.7

# ROC CURVE
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob) # 0.8647228131648104

# Classification report
print(classification_report(y_test, y_pred))

#                  precision    recall  f1-score   support
#            0       0.86      0.87      0.86       131
#            1       0.71      0.69      0.70        61
#     accuracy                           0.81       192
#    macro avg       0.78      0.78      0.78       192
# weighted avg       0.81      0.81      0.81       192
