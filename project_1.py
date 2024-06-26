# -*- coding: utf-8 -*-
"""Project 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ybNDB2wzJgzg_feJODiiNsimUg8GsSwy

# Domain proyek

# Data Loading
"""

# Commented out IPython magic to ensure Python compatibility.
# Import library yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load dataset
dataset = "/content/ObesityDataSet.csv"
df = pd.read_csv(dataset)
df

df.info()

"""# Exploratory Data Analysis - Deskripsi Variabel

## Deskripsi Variabel
"""

# Bulatkan nilai pada kolom "Age" ke bawah
df['Age'] = df['Age'].astype(int)

# Verifikasi perubahan
df

"""Pada kolom "Age", Dtype-nya adalah Float, sedangkan umur seharusnya adalah int. Maka, jenis datanya akan dirubah dan kolom tersebut akan dibulatkan ke bawah."""

df.info()

df.describe()

df.isna().sum()
#outlier

"""Tidak terdapat _missing Value_ pada dataset ini dan setelah melihat `df.describe()`, data sudah cukup baik dan dapat segera diproses.

# Data Preparation 1
"""

# Buat salinan dataset awal
dfe = df.copy()

# Menggunakan LabelEncoder untuk mengkodekan variabel objek
label_encoder = preprocessing.LabelEncoder()

# Daftar kolom yang ingin diencode
columns_to_encode = ['Gender','family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

# Loop melalui setiap kolom dan mengkodekannya
for col in columns_to_encode:
    dfe[col] = label_encoder.fit_transform(dfe[col])

# Menampilkan hasil
print(dfe[columns_to_encode].head())
dfe.head()

dfe.corr()

# Menghapus kolom-kolom yang disebutkan
columns_to_drop = ['Gender', 'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
dfe.drop(columns=columns_to_drop, inplace=True)

# Menampilkan DataFrame setelah menghapus kolom-kolom
print(dfe.head())

dfe.info()

"""Sekarang, dataset `dfe` hanya tersisa dataset yang kita perlukan saja untuk dianalisis lebih lanjut

# Exploratory Data Analysis-Univariate Analysis

Bagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features.
"""

categorical_features = ['family_history_with_overweight', 'CAEC']
numerical_features = ['Age','Weight','Height']

"""## Categorical Features"""

feature = categorical_features[0]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
dfu = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(dfu)
count.plot(kind='bar', title=feature);

feature = categorical_features[1]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
dfu = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(dfu)
count.plot(kind='bar', title=feature);

"""## Numerical Features"""

df.hist(bins=50, figsize=(20,15))
plt.show()

"""# Data Preparation 2

## Reduksi Dimensi dengan PCA
"""

sns.pairplot(df[['Height','Weight']], plot_kws={"s": 2});

pca = PCA(n_components=2, random_state=123)
pca.fit(df[['Height','Weight']])
princ_comp = pca.transform(dfe[['Height','Weight']])

pca.explained_variance_ratio_.round(4)

pca = PCA(n_components=1, random_state=123)
pca.fit(dfe[['Height','Weight']])
dfe['dimension'] = pca.transform(dfe.loc[:, ('Height','Weight')]).flatten()
dfe.drop(['Height','Weight'], axis=1, inplace=True)

"""## Train-Test-Split"""

X = dfe.drop(["NObeyesdad"],axis =1)
y = dfe["NObeyesdad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""## Standarisasi"""

numerical_features = ['Age','dimension']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

"""# Model Development"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""### KNN"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""### Random Forest"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor

# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""### Boosting Alg"""

from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""# Evaluasi Model"""

# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))

# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""Dari gambar di atas, terlihat bahwa, model Random Forest (RF) memberikan nilai eror yang paling kecil. Sedangkan model dengan algoritma Boosting memiliki eror yang paling besar (berdasarkan grafik, angkanya di atas 2.0). Sehingga model RF yang akan kita pilih sebagai model terbaik untuk melakukan prediksi harga diamonds."""

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

"""Terlihat bahwa prediksi dengan Random Forest (RF) dan KNN memberikan hasil yang sama dan prediksi_Boosting cukup mendekati."""