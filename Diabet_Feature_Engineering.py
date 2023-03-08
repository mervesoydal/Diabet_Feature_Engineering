#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

#############################################
# İş Problemi
#############################################
""""
Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine
öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve
özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.
"""

#############################################
# Veri Seti Hikayesi
#############################################

"""
Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin 
parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde 
olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. Hedef değişken 
"outcome" olarak belirtilmiş olup; 1diyabet test sonucunun pozitif oluşunu, 0 ise negatifoluşunu belirtmektedir.
"""

# Proje Görevleri:

#STEP 1: Keşifçi Veri Analizi

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import  LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv("datasets/diabetes.csv")
    return data


df = load()
df.head()
df.shape
df.info()

"""
Pregnancies: Hamilelik sayısı 
Glucose Oral: glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu 
Blood Pressure: Kan Basıncı (Küçüktansiyon) (mm Hg) 
Skin Thickness: Cilt Kalınlığı 
Insulin: 2 saatlik serum insülini (mu U/ml) 
Diabetes Pedigree Function: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu) 
BMI: Vücut kitle endeksi 
Age: Yaş (yıl) 
Outcome: Hastalığasahip(1) ya da değil(0)

"""


# STEP 1.2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 768
Variables: 9
cat_cols: 1
num_cols: 8
cat_but_car: 0
num_but_cat: 1
"""
#cat_cols = ['Outcome']
#num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
# 'DiabetesPedigreeFunction', 'Age']

# STEP 1.3:Kategorik değişkenlerin analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome")

for col in cat_cols:
    cat_summary(df, col, plot=False)

"""

   Outcome  Ratio
0      500 65.104
1      268 34.896
##########################################

"""

#Numerik değişkenlerin analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)

#TASK 1.4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
#hedef değişkene göre numerik değişkenlerin ortalaması)

def target_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}))

for col in num_cols:
 target_with_num(df,"Outcome",col)


# TASK 1.6: KORELASYON ANALİZİ
#korelasyon matrisi

df.corr()

#Matrisi Görselleştirme
sns.heatmap(df.corr(),
    annot=True, fmt=".2g")
plt.show()

#Eksik Değer Analizi

df.head()

#Pregnancies ve Outcome haricindeki değerlerde 0 bulunmaması gerekir. Veri setinde eksik gözlem bulunmamakta
# ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Öncelikle bunu gözlemleyelim.
# eksik gozlem var mı yok mu sorgusu
df.isnull().sum()

"""
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0

"""

df.isnull().values.any()
#False

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

# zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
 df[col] = np.where(df[col] == 0 , np.nan, df[col])

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

"""
               n_miss  ratio
Insulin           374 48.700
SkinThickness     227 29.560
BloodPressure      35  4.560
BMI                11  1.430
Glucose             5  0.650

"""

# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_cols)

#Eksik değerler saptandıktan sonra nasıl dolduracağımıza bakalım.
df.isnull().sum()

"""
Pregnancies                   0
Glucose                       5
BloodPressure                35
SkinThickness               227
Insulin                     374
BMI                          11
DiabetesPedigreeFunction      0
Age                           0
Outcome                       0
"""

for col in zero_columns:
 df.loc[df[col].isnull(), col] = df[col].median()

"""
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
"""

#TASK 1.5: AYKIRI DEĞER ANALİZİ
# 1. Eşik değer belirleme
# 2. Aykırılara eriştik.
# 3. Aykırı değer var mı yok diye bakıldı.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#Example:
outlier_thresholds(df, "Age")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")
check_outlier(df, "SkinThickness")
df.head()

# Outlier değerleri tespit ettikten sonra threshould değerleri ile değiştiriyoruz.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in df.columns:
    print(col, check_outlier(df, col))

""""
Pregnancies True
Glucose False
BloodPressure True
SkinThickness True
Insulin True
BMI True
DiabetesPedigreeFunction True
Age True
Outcome False
"""
for col in df.columns:
    replace_with_thresholds(df, col)

# "replace_with_thresholds" uyguladıktan sonra:
"""
Pregnancies False
Glucose False
BloodPressure False
SkinThickness False
Insulin False
BMI False
DiabetesPedigreeFunction False
Age False

"""
# Yeni değişkenler oluşturalım.

# Yaş değişkenini kategorilere ayırıp yeni özellikli gruplar oluşturalım.

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = "young"
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = "mature"
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = "senior"

df.head(5)

#BMI_NEW değişkeni

df['BMI_NEW'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 40, 100],
                       labels=["Underweight", "Normal", "Overweight", "Obese", "OverObese"])

df['BMI_NEW'].apply(lambda x: float(x))

df['BMI_NEW'] = pd.to_numeric(df['BMI_NEW'], errors = 'coerce')

df.info()
df.dtypes
# Glukoz değeri ile yeni GLUCOSE_NEW değişkeni

df["GLUCOSE_NEW"] = pd.cut(x=df["Glucose"], bins=[0, 70, 120, 140, 200],
                           labels=["Not Normal", "Normal", "Prediabete", "Diabete"])

# Age ile BMI değişkeni arasındaki ilişkiden yola çıkarak yeni bir değişken üretelim.

df.loc[(df['Age'] < 18) & (df['BMI'] < 18.5), 'NEW_AGE_BMI'] = "Underweightyoung"
df.loc[(df['BMI'] < 18.5) & (df['Age'] > 18) & (df['Age'] <= 56), 'NEW_AGE_BMI'] = "Underweightmature"
df.loc[(df['Age'] >= 56) & (df['BMI'] < 18.5) , 'NEW_AGE_CAT'] = "Underweightsenior"

df.loc[(df['Age'] < 18) & (df['BMI'] > 18.5) & (df['BMI'] <= 24.9) ,'NEW_AGE_BMI'] = "normalyoung"
df.loc[(df['Age'] >= 18) & (df['Age'] < 56) & (df['BMI'] > 18.5) & (df['BMI'] <= 24.9) ,'NEW_AGE_BMI'] = "normalmature"
df.loc[(df['Age'] >= 56) & (df['BMI'] > 18.5) & (df['BMI'] <=  24.9) ,'NEW_AGE_BMI'] = "normalsenior"

df.loc[(df['Age'] < 18) & (df['BMI'] > 24.9) & (df['BMI'] <= 29.9) ,'NEW_AGE_BMI'] = "overweightyoung"
df.loc[(df['Age'] >= 18) & (df['Age'] < 56) & (df['BMI'] > 24.9) & (df['BMI'] <= 29.9) ,'NEW_AGE_BMI'] = "overweightmature"
df.loc[(df['Age'] >= 56) & (df['BMI'] > 24.9) & (df['BMI'] <= 29.9) ,'NEW_AGE_BMI'] = "overweightsenior"

df.loc[(df['Age'] < 18) & (df['BMI'] > 29.9) & (df['BMI'] <= 40) ,'NEW_AGE_BMI'] = "obeseyoung"
df.loc[(df['Age'] >= 18) & (df['Age'] < 56) & (df['BMI'] > 29.9) & (df['BMI'] <= 40) ,'NEW_AGE_BMI'] = "obesemature"
df.loc[(df['Age'] >= 56) & (df['BMI'] > 29.9) & (df['BMI'] <= 40) ,'NEW_AGE_BMI'] = "obesesenior"

df.loc[(df['Age'] < 18) & (df['BMI'] > 40) & (df['BMI'] <= 100) ,'NEW_AGE_BMI'] = "overobeseyoung"
df.loc[(df['Age'] >= 18) & (df['Age'] < 56) & (df['BMI'] > 40) & (df['BMI'] <= 100) ,'NEW_AGE_BMI'] = "overobesemature"
df.loc[(df['Age'] >= 56) & (df['BMI'] > 40) & (df['BMI'] <= 100) ,'NEW_AGE_BMI'] = "overobesesenior"


df.head()
# Insulin değeri ile yeni INSULIN_NEW değişkeni

df['INSULIN_NEW'] = pd.cut(x = df['Insulin'], bins = [0, 100, 126, 319], labels=["Not Healthy", "Healthy", "Diabete"])
df['Insulin'].sort_values(ascending = False)

# Blood Pressure ile yeni Blood_Pressure_New değişkeni

df['BloodPressure_New'] = pd.cut(x=df['BloodPressure'], bins=[0, 80, 85, 89, 200], labels=["Not Normal", "Normal", "High", "Very High"])

# ENCODING
cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 768
Variables: 15
cat_cols: 7
num_cols: 8
cat_but_car: 0
num_but_cat: 5

"""

#Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

# binary_cols
# ['NEW_AGE_CAT']

for col in binary_cols:
    label_encoder(df, col)

df.head()

# One hot encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

new_cat_cols = [col for col in cat_cols if col not in binary_cols and col not in  ["Outcome"]]
df.dtypes

#new_cat_cols
#['BloodPressure_New', 'INSULIN_NEW', 'BMI_NEW', 'GLUCOSE_NEW', 'NEW_AGE_BMI']

df = one_hot_encoder(df, new_cat_cols, drop_first=True).head()
df.head()
# Encoding işleminden sonra Standartlaştırma işlemine geçelim.
# Numerik değişkenler için ortalamayı 0 standart sapması 1 olacak şekilde standartlaşma işlemi yapacağız.

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()

#num_cols
#['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

"""
 Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI  DiabetesPedigreeFunction    Age  Outcome  NEW_AGE_CAT     BMI_NEW GLUCOSE_NEW        NEW_AGE_BMI INSULIN_NEW BloodPressure_New
0        0.647    0.866         -0.031          0.825    0.039  0.181                     0.589  1.446    1.000            0       Obese     Diabete       Obese_Mature     Healthy        Not Normal
1       -0.849   -1.205         -0.544          0.018    0.039 -0.869                    -0.378 -0.189    0.000            0  Overweight      Normal  Overweight_Mature     Healthy        Not Normal
2        1.246    2.017         -0.715          0.018    0.039 -1.365                     0.747 -0.103    1.000            0      Normal     Diabete      Normal_Mature     Healthy        Not Normal
3       -0.849   -1.074         -0.544         -0.789   -1.494 -0.644                    -1.023 -1.050    0.000            0  Overweight      Normal  Overweight_Mature     Healthy        Not Normal
4       -1.148    0.504         -2.768          0.825    1.414  1.607                     2.597 -0.017    1.000            0  Over Obese  Prediabete   Overobese_Mature     Diabete        Not Normal

"""


# Son aşama ise Model oluşturmaktır.

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=18)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=40).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
0