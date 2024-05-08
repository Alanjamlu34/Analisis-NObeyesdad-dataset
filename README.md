# Laporan Proyek Machine Learning - Paulinus Alan Sanjaya Jamlu
## Domain Proyek
Jumlah penderita obesitas di seluruh dunia lebih dari satu miliar orang. Berdasarkan penelitian yang dipublikasikan dalam jurnal The Lancet, data dari tahun 2022 hingga tahun terakhir yang dimasukkan dalam analisis menunjukkan bahwa ada 879 juta orang dewasa dan 159 juta anak-anak yang mengalami obesitas atau kelebihan berat badan.

Secara global, angka obesitas di kalangan anak-anak dan remaja meningkat empat kali lipat dari tahun 1990 hingga 2022, sementara angka obesitas di kalangan orang dewasa meningkat lebih dari dua kali lipat. Penelitian ini juga mengungkapkan bahwa tingkat anak-anak dan remaja yang mengalami kekurangan berat badan menurun pada periode yang sama, dan penurunannya lebih dari separuh di antara orang dewasa di seluruh dunia. Oleh karena itu, obesitas saat ini menjadi bentuk malnutrisi yang paling umum di banyak negara.
Dari masalah tersebut, menggunakan dataset Obesity Levels & Life Style dari kagle, dapat dilihat aspek mana saja yang memberikan pengaruh besar pada tingkat obesitas sehingga dapat dihindari sehingga dapat melakukan pencegahan dini.

Pada tahun 2008, sekitar 2,8 juta orang dewasa meninggal akibat obesitas, sekitar 300 juta orang yang secara klinis tergolong obesitas yang merupakan penyokong utama penyakit degeneratif seperti diabetes, penyakit jantung, dan kanker[1]. Dari dataset Obesity Levels & Life Style, _machine learning_ dapat digunakan untuk menganalisis faktor-faktor yang paling berpengaruh dalam peningkatan obesitas seseorang dan model tersebut juga dapat digunakan untuk memprediksi data yang serupa kedepannya.



## Business Understanding
### Problem Statements
- Variabel apa yang paling mempengaruhi tingkat obesitas seseorang?
- Dapatkah model _machine learning_ mengklasifikasikan tingkat obesitas seseorang dengan data baru?

### Goals
- Menganalisis pengaruh setiap variabel dan mengeliminasi variabel yang tidak berpengaruh atau pengaruhnya tidak signifikan terhapat perubahan tingkat obesitas. Signifikan yang dimaksud adalah, apakah variabel tersebut berpengaruh pada tingkat obesitas atau tidak atau seberapa besar perubahan obesitas jika variabel tersebut berubah.
- Menggunakan model yang tersedia melalui proses analisis untuk mengklasifikasikan kategori obesitas yang ada berdasarkan model yang telah dibuat.

## Data Understanding
Dataset ini adalah dataset yang diambil dari [Kagle]( https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels) berdasarkan jurnal dari Fabio MendozaPalecho dan Alexis de la Hoz Manotas yang bertujuan untuk meneliti pengaruh gaya hidup dan beberapa faktor lain terhadap obesitas di negara Meksiko[2], Peru dan Colombia dengan jumlah sample 2111 dan 17 atribut.

Download dataset   : https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels

Variabel/atribut pada dataset ini adalah sebagai berikut:
- Gender (Jenis Kelamin): Menyatakan jenis kelamin individu (misalnya, “Laki-laki” atau “Perempuan”).
- Age (Usia): Menyatakan usia individu dalam tahun.
- Height (Tinggi Badan): Menyatakan tinggi badan individu dalam satuan tertentu (misalnya, sentimeter).
- Weight (Berat Badan): Menyatakan berat badan individu dalam satuan tertentu (misalnya, kilogram).
- Family History with Overweight (Riwayat Keluarga dengan Kelebihan Berat Badan): Menyatakan apakah individu memiliki anggota keluarga yang menderita atau pernah menderita kelebihan berat badan.
- Frequency of High-Calorie Food Consumption (Frekuensi Konsumsi Makanan Kaya Kalori): Menyatakan seberapa sering individu mengonsumsi makanan kaya kalori.
- Frequency of Vegetable Consumption (Frekuensi Konsumsi Sayuran): Menyatakan seberapa sering individu mengonsumsi sayuran dalam makanan mereka.
- Number of Main Meals Per Day (Jumlah Makan Utama Harian): Menyatakan berapa kali individu makan utama dalam sehari.
- Between-Meal Food Consumption (Konsumsi Makanan Antara Makan Utama): Menyatakan apakah individu mengonsumsi makanan di antara makan utama.
- Smoking Status (Merokok): Menyatakan apakah individu merokok.
- Daily Water Intake (Konsumsi Air): Menyatakan berapa banyak air yang diminum individu setiap hari.
- Daily Calorie Monitoring (Pemantauan Kalori Harian): Menyatakan apakah individu memantau kalori yang dikonsumsi setiap hari.
- Physical Activity Frequency (Aktivitas Fisik): Menyatakan seberapa sering individu melakukan aktivitas fisik.
- Technology Use (Penggunaan Perangkat Teknologi): Menyatakan berapa lama individu menggunakan perangkat teknologi seperti ponsel, video game, televisi, komputer, dan lainnya.
- Alcohol Consumption (Konsumsi Alkohol): Menyatakan seberapa sering individu mengonsumsi alkohol.
- Transportation Mode (Transportasi yang Digunakan): Menyatakan jenis transportasi yang biasanya digunakan oleh individu.
- Obesity Level (Tingkat Obesitas): Menyatakan tingkat obesitas individu (misalnya, “Kurang Berat Badan”, “Normal”, “Obesitas Kelas I”, dll.).

## Data Preparation
Berikut adalah teknik-teknik yang digunakan pada tahap ini:
- Encoding: Proses ini mengubah variabel dengan `Dtype: Object` menjadi `float` dengan mengubah nilainya (merepresentasikan) menjadi angka agar dapat dianalisis korelasinya.
- Correlation: Proses ini bertujuan untuk melihat korelasi/hubungan setiap variabel dengan variabel NObeyesdad. Syarat variabell signifikan adalah corr>0.2 atau corr<-0.2. Jika tidak memenuhi syarat tersebut akan dieliminasi/dihapus. Berikut adalah data korelasi antara NObeyesdad dengan 16 variabel lain. 

|             |   Gender |      Age |   Height |   Weight | family_history_with_overweight |
|-------------|---------:|---------:|---------:|---------:|-------------------------------:|
| NObeyesdad | 0.024908 | 0.229053 | 0.038986 | 0.387643 | 0.313667                       |

---


|             |     FAVC |     FCVC |       NCP |     CAEC |     SMOKE |     CH2O |       SCC |
|-------------|---------:|---------:|----------:|---------:|----------:|---------:|----------:|
| Noobeyesdad | 0.044582 | 0.018522 | -0.092616 | 0.327295 | -0.023256 | 0.108868 | -0.050679 |

---
|            |       FAF |       TUE |      CALC |    MTRANS |     SMOKE |     CH2O |       SCC |
|------------|----------:|----------:|----------:|----------:|----------:|---------:|----------:|
| NObeyesdad | -0.129564 | -0.069448 | -0.134632 | -0.046202 | -0.023256 | 0.108868 | -0.050679 |

Tabel 1. Tabel korelasi NObeyesdad

Dari tabel di atas, berikut adalah variabel yang tidak/kurang berpengaruh pada variabel NObeyesdad menurut syarat:

`['Gender', 'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']`.

 Jika dilihat, Variabel "Height" memiliki korelasi `0.2< corr_Height >-0.2` namun tidak dieliminasi. Hal ini dikarenakan dalam melihat tingkat obesitas seseorang, rasio tinggi badan dan berat badan sangat penting untuk dilihat meskipun tidak ada korelasi antar keduanya. Sederhananya, semakin tinggi seseorang tidak berarti dia semakin gemuk (obesitas), tetapi "secara sederhana" perbandingan antar berat badan dan tinggilah yang akan menentukannya.

- Dimention Reduction with PCA: Proses ini mereduksi variabel 'Height' dan 'Weight' menjadi satu variabel dengan nama 'dimensi'.

- Train-Test-Split: Membagi dataset menjadi data latih (train) dan data uji (test) dengan proporsi 80:20. Hasilnya adalah:

  Total # of sample in whole dataset: 2111

  Total # of sample in train dataset: 1899

  Total # of sample in test dataset: 21

- Standaritation: Proses standarisasi menggunakan `Standardscaler()`: Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

## Model Development
Pada tahap ini, akan dikembangkan model machine learning dengan tiga algoritma. Kemudian, akan dilakukan evaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Nilai dari parameter pada ketiga algoritma ini diambil dengan pertimbangan _running time_ dari algoritma dan akurasinya. Setelah mencoba berbagai nilai parameter, didapat nilai parameter yang dirasa sudah cukup baik sesuai dengan yang digunakan dalam algoritma berikut.
Ketiga algoritma yang akan digunakan, antara lain:
  -  _K-Nearest Neighbor_
  - *RFF*
  - _Boosting Algorithm_

1. _K-Nearest Neighbor_
   
   _KNN_ adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma _KNN_ menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. k = 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik, lalu melatih data training dan menyimpan data testing untuk tahap evaluasi.
   Meskipun algoritma _KNN_ mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai curse of dimensionality (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data.

2. *RFF*

   *RFF* adalah sebuah metode ensemble yang menggabungkan beberapa decision tree regressors untuk memprediksi nilai kontinu (misalnya, harga rumah, suhu, dll.).
   Setiap decision tree dalam forest memproses sub-sampel dari dataset dan menghasilkan prediksi. *RFF* mengatasi masalah overfitting yang sering terjadi pada single decision tree. Akhirnya, hasil prediksi dari semua trees digabungkan (dengan cara rata-rata) untuk menghasilkan prediksi akhir. Setelah import library yang diperlukan, Selanjutnya, membuat objek model RF dengan menginstansiasi RandomForestRegressor. 
   Berikut beberapa parameter yang diatur yaitu n_estimators= 50.Jumlah trees dalam forest terdiri dari 50 pohon. Semakin banyak trees, semakin baik performanya, tetapi juga semakin lambat komputasinya. Parameter max_depth= 16 merupakan maksimum kedalaman setiap tree. Artinya, setiap pohon akan memiliki kedalaman maksimum 16. Parameter random_state= 55 merupakan parameter yang memastikan hasil yang konsisten setiap kali model dijalankan dan nilai -1 pada parameter n_jobs= -1 menunjukkan bahwa semua CPU yang tersedia akan digunakan untuk menghitung pohon-pohon dalam *RFF* secara paralel. Ini akan mempercepat proses pelatihan model.Setelah nilai parameter ditentukan, model dilatih dengan memanggil metode .fit(X_train, y_train), di mana X_train adalah fitur dan y_train adalah target (label) dari data pelatihan. Akhirnya, dihitung _mean squared error_(MSE) pada data pelatihan dan menyimpannya di dalam DataFrame models pada baris train_mse dan kolom RandomForest.

3. _Boosting Algorithm_
    
    AdaBoostRegressor adalah algoritma ensemble yang digunakan untuk memperbaiki performa model regresi. Berikut adalah penjelasan singkat mengenai AdaBoostRegressor: AdaBoostRegressor adalah metode*RFF* yang menggabungkan beberapa model regresi sederhana (biasanya decision trees) menjadi satu model yang lebih kuat. Algoritma ini bekerja dengan menggabungkan hasil dari model-model yang lemah untuk menghasilkan prediksi yang lebih akurat. Langkah awal adalah dengan inisiasi Model: Selanjutnya, dibuatkan objek model *RFF* dengan menginstansiasi AdaBoostRegressor. Beberapa parameter yang diatur seperti learning_rate= 0.05: Nilai ini mengontrol seberapa besar kontribusi setiap model lemah terhadap model akhir. Semakin kecil nilai ini, semakin lambat konvergensi model. random_state= 55: Parameter ini menentukan inisialisasi random state untuk memastikan hasil yang konsisten pada setiap eksekusi. Setelah menetapkan nilai parameter, model dilatih dengan memanggil metode .fit(X_train, y_train), di mana X_train adalah fitur dan y_train adalah target (label) dari data pelatihan. Akhirnya, dihitung _mean squared error_ (MSE) pada data pelatihan dan menyimpannya di dalam DataFrame models pada baris ‘train_mse’ dan kolom ‘ *RFF*’.


## Evaluasi Model
Metrik yang digunakan adalah _Mean Squarre Error_ (MSE) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^n (y_i - \hat{y}_{\text{pred}_i})^2
$$

_Keterangan:_

*N = jumlah dataset*

*y_i* = nilai sebenarnya_

*y_pred = nilai prediksi*

Setelah melakukan perhitungan metriks dengan MSE, hasilnya adalah sebagai berikut:

|          | train    | test     |
|----------|----------|----------|
|  *KNN*     | 1.261564 | 1.522972 |
| *RFF*      | 0.202345 | 1.202889 |
|*RFF* | 1.856123 | 2.145006 |

Tabel 2. Hasil metriks MSE
      
Perhatikan plot berikut sebagai representasi dari data di atas
  ![image](https://github.com/Alanjamlu34/Analisis-NObeyesdad-dataset/assets/142156489/0218dbb5-d2ba-484a-a4b6-0a3b6d9b9774)

Gambar 1. Plot MSE train-test

Dari gambar di atas, terlihat bahwa, model *Random Forest* (*RFF*) memberikan nilai eror yang paling kecil. Sedangkan model dengan algoritma *RFF* memiliki eror yang paling besar (berdasarkan grafik, angkanya di atas 2.0). Sehingga model *RFF* yang akan dipilih sebagai model terbaik untuk melakukan prediksi tingkat obesitas.
  |     | y_true | prediksi_*KNN*| prediksi_*RFF* | prediksi_Bossting |
  |-----|--------|--------------|-------------|-------------------|
  | 600 | 0      | 0.0          | 0.0         | 0.1               |
  
  Tabel 3. Hasil prediksi

1. y_true: Nilai ini menunjukkan target aktual (nilai sebenarnya) dari data uji pada baris ke-600. Dalam kasus ini, nilai target aktual adalah 0.
2. prediksi_*KNN*, prediksi_*RF*, dan prediksi_*boosting*: Nilai-nilai ini adalah hasil prediksi dari masing-masing model pada baris ke-600. Model-model yang digunakan:
    - _KNN_ (*K-Nearest Neighbors*): Nilai prediksi dari model _KNN_ adalah 0.0.
    - *RFF*: Nilai prediksi dari model *RFF* juga adalah 0.0.
    - AdaBoost: Nilai prediksi dari model AdaBoost adalah 0.1.
        
Pada baris ke-600, ketiga model memprediksi nilai target yang rendah (sekitar 0). Semua model tampaknya setuju bahwa nilai target sebenarnya adalah 0. Dari hasil perhitungan MSE, berdasarkan plot dan hasil prediksi data test, dapat disimpulkan bahwa algoritma *RFF* merupakan algoritma yang memiliki hasil paling optimal diikuti oleh _KNN_ dan terakhir adalah _Boosting Algorithm_.

Jadi, hasil analisis mendapatkan 5 variabel yang berpengaruh pada tingkat obesitas seseorang yaitu umuir, berat badan, riwayat keluarga penderita obesitas dan konsumsi makanan selain makanan utama. Model juga dapat meprediksi tingkat obesitas dengan sangat baik sehingga model ini dapat digunakan untuk memprediksi tingkat obesitas dari data yang baru.


## Daftar Pustaka
[1] Widiantini, Winne dan Zarfiel Tafal. “Aktivitas Fisik, Stres, dan Obesitas pada Pegawai Negeri Sipil.” _Jurnal Kesehatan Masyarakat Nasional (Kesmas)_, Vol. 8, No. 7, 2014.

[2] Palechor, Fabio Mendoza, dan Alexis de la Hoz Manotas. “_Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru, and Mexico._” Universidad de la Costa, CUC, Colombia, Vol. 25, 2019.



