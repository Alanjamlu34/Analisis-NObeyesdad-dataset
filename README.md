# Laporan Proyek Machine Learning - Paulinus Alan Sanjaya Jamlu
## Domain Proyek
Jumlah penderita obesitas di seluruh dunia lebih dari satu miliar orang. Berdasarkan penelitian yang dipublikasikan dalam jurnal The Lancet, data dari tahun 2022 hingga tahun terakhir yang dimasukkan dalam analisis menunjukkan bahwa ada 879 juta orang dewasa dan 159 juta anak-anak yang mengalami obesitas atau kelebihan berat badan.

Secara global, angka obesitas di kalangan anak-anak dan remaja meningkat empat kali lipat dari tahun 1990 hingga 2022, sementara angka obesitas di kalangan orang dewasa meningkat lebih dari dua kali lipat. Penelitian ini juga mengungkapkan bahwa tingkat anak-anak dan remaja yang mengalami kekurangan berat badan menurun pada periode yang sama, dan penurunannya lebih dari separuh di antara orang dewasa di seluruh dunia. Oleh karena itu, obesitas saat ini menjadi bentuk malnutrisi yang paling umum di banyak negara.
Dari masalah tersebut, menggunakan dataset Obesity Levels & Life Style dari kagle, dapat dilihat aspek mana saja yang memberikan pengaruh besar pada tingkat obesitas sehingga dapat dihindari dan anda dapat melihat apakah anda mengalami obesitas atau tidak berdasarkan hasil analisis dari dataset tersebut.

Dikutip dari jurnal yang ditulis oelh Winne Widiantini, pada tahun 2008, sekitar 2,8 juta orang dewasa meninggal akibat obesitas, sekitar 300 juta orang yang secara klinis tergolong obesitas yang merupakan penyokong utama penyakit degeneratif seperti diabetes, penyakit jantung, dan kanker. Dari dataset Obesity Levels & Life Style, _machine learning_ dapat digunakan untuk menganalisis faktor-faktor yang paling berpengaruh dalam peningkatan obesitas seseorang dan model tersebut juga dapat digunakan untuk memprediksi data yang serupa kedepannya.



## Business Understanding
### Problem Statements
- Variabel apa yang paling mempengaruhi tingkat obesitas seseorang?
- Dapatkan model _machine learning_ mengklasifikasikan tingkat obesitas seseorang dengan data baru?

### Goals
- Menganalisis pengaruh setiap variabel dan mengeliminasi variabel yang tidak berpengaruh atau pengaruhnya tidak signifikan terhapat perubahan tingkat obesitas. Signifikan yang dimaksud adalah, apakah variabel tersebut berpengaruh pada tingkat obesitas atau tidak atau seberapa besar perubahan obesitas jika variabel tersebut berubah.
- Menggunakan model yang tersedia melalui proses analisis untuk mengklasifikasikan kategori obesitas yang ada berdasarkan model yang telah dibuat.

## Data Understanding
Dataset ini adalah dataset yang diambil dari [Kagle]( https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels) berdasarkan jurnal dari Fabio MendozaPalecho dan Alexis de la Hoz Manotas yang bertujuan untuk meneliti pengaruh gaya hidup dan beberapa faktor lain terhadap obesitas di negara Meksiko, Peru dan Colombia dengan jumlah sample 2111 dan 17 atribut.

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
- Correlation: Proses ini bertujuan untuk melihat korelasi/hubungan setiap variabel dengan variabel NObeyesdad. Akan diambil korelasi yang cukup signifikan (corr>0.2 atau corr<-0.2) dan mengeliminasi atribut yang tidak memenuhi kriteria tersebut. Berikut adalah variabel yang tidak/kurang berpengaruh pada variabel NObeyesdad: `['Gender', 'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']`. Jika dilihat, Variabel "Height" memiliki korelasi <0.2 dan >-0.2 namun tidak dieliminasi. Hal ini dikarenakan dalam melihat tingkat obesitas sesorang, rasio tinggi badan dan berat badan sangat penting untuk dilihat meskipun tidak ada korelasi antar keduanya. Sederhananya, semakin tinggi seseorang tidak berarti dia semakin gemuk (obesitas), tetapi "secara sederhana" perbandingan antar berat badan dan tinggilah yang akan menentukannya.
- Dimention Reduction with PCA: Proses ini mereduksi variabel 'Height' dan 'Weight' menjadi satu variabel dengan nama 'dimensi'.

- Train-Test-Split: Membagi dataset menjadi data latih (train) dan data uji (test) dengan proporsi 80:20. Hasilnya adalah:\
  Total # of sample in whole dataset: 2111\
  Total # of sample in train dataset: 1899\
  Total # of sample in test dataset: 21

- Standaritation: Proses standarisasi menggunakan `Standardscaler()`: Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

## Model Development
Pada tahap ini, akan dikembangkan model machine learning dengan tiga algoritma. Kemudian, akan dilakukan evaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, antara lain:
  - K-Nearest Neighbor
  - Random Forest
  - Boosting Algorithm
1. K-Nearest Neighbor
   
   KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.
   `k = 10` tetangga dan metric Euclidean untuk mengukur jarak antara titik, lalu melatih data training dan menyimpan data testing untuk tahap evaluasi.
   Meskipun algoritma KNN mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai curse of dimensionality (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data.

3. Random Forest

   Random Forest adalah sebuah metode ensemble yang menggabungkan beberapa decision tree regressors untuk memprediksi nilai kontinu (misalnya, harga rumah, suhu, dll.).
   Setiap decision tree dalam forest memproses sub-sampel dari dataset dan menghasilkan prediksi.
   Akhirnya, hasil prediksi dari semua trees digabungkan (dengan cara rata-rata) untuk menghasilkan prediksi akhir.
   Random Forest mengatasi masalah overfitting yang sering terjadi pada single decision tree.
   Parameter dalam Kode:
    - `n_estimators= 50`: Jumlah trees dalam forest terdiri dari 50 pohon. Semakin banyak trees, semakin baik performanya, tetapi juga semakin lambat komputasinya.
    - `max_depth= 16`: Maksimum kedalaman setiap tree. Artinya, setiap pohon akan memiliki kedalaman maksimum 16.
    - `random_state= 55`: Nilai ini memastikan hasil yang konsisten setiap kali model dijalankan.
    - `n_jobs= -1` : Nilai -1 pada parameter ini menunjukkan bahwa semua CPU yang tersedia akan digunakan untuk menghitung pohon-pohon dalam random forest secara paralel. Ini akan mempercepat proses pelatihan model.
  4. Boosting Algorithm

     AdaBoostRegressor adalah algoritma ensemble yang digunakan untuk memperbaiki performa model regresi. Berikut adalah penjelasan singkat mengenai AdaBoostRegressor:
     AdaBoostRegressor adalah metode boosting yang menggabungkan beberapa model regresi sederhana (biasanya decision trees) menjadi satu model yang lebih kuat.
    Algoritma ini bekerja dengan menggabungkan hasil dari model-model yang lemah untuk menghasilkan prediksi yang lebih akurat.

     - `learning_rate= 0.05`: Nilai ini mengontrol seberapa besar kontribusi setiap model lemah terhadap model akhir. Semakin kecil nilai ini, semakin lambat konvergensi model.
     - `random_state= 55`: Parameter ini menentukan inisialisasi random state untuk memastikan hasil yang konsisten pada setiap eksekusi.

## Evaluasi Model
Metrik yang digunakan adalah Mean Squarre Error (MSE) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^n (y_i - \hat{y}_{\text{pred}_i})^2
$$

_Keterangan:_

_N = jumlah dataset_

_y_i = nilai sebenarnya_

_y_pred = nilai prediksi_

Setelah melakukan perhitungan metriks dengan MSE, hasilnya adalah sebagai berikut

|          | train    | test     |
|----------|----------|----------|
| KNN      | 1.261564 | 1.522972 |
| RF       | 0.202345 | 1.202889 |
| Boosting | 1.856123 | 2.145006 |
Tabel 1. Hasil metriks MSE
      
Perhatikan plot berikut sebagai representasi dari data di atas
  ![image](https://github.com/Alanjamlu34/Analisis-NObeyesdad-dataset/assets/142156489/0218dbb5-d2ba-484a-a4b6-0a3b6d9b9774)\
Gambar 1. Plot MSE train-test

Dari gambar di atas, terlihat bahwa, model _Random Forest_ (RF) memberikan nilai eror yang paling kecil. Sedangkan model dengan algoritma _Boosting_ memiliki eror yang paling besar (berdasarkan grafik, angkanya di atas 2.0). Sehingga model RF yang akan dipilih sebagai model terbaik untuk melakukan prediksi tingkat obesitas.
  |     | y_true | prediksi_KNN | prediksi_RF | prediksi_Bossting |
  |-----|--------|--------------|-------------|-------------------|
  | 600 | 0      | 0.0          | 0.0         | 0.1               |
  
  Tabel 2. Hasil prediksi

1. y_true:
    - Nilai ini menunjukkan target aktual (nilai sebenarnya) dari data uji pada baris ke-600. Dalam kasus ini, nilai target aktual adalah 0.
2. prediksi_KNN, prediksi_RF, dan prediksi_Boosting:
    - Nilai-nilai ini adalah hasil prediksi dari masing-masing model pada baris ke-600.
    - Model-model yang digunakan:
      - KNN (K-Nearest Neighbors): Nilai prediksi dari model KNN adalah 0.0.
      - Random Forest: Nilai prediksi dari model Random Forest juga adalah 0.0.
      - AdaBoost: Nilai prediksi dari model AdaBoost adalah 0.1.
        
Pada baris ke-600, ketiga model memprediksi nilai target yang rendah (sekitar 0). Semua model tampaknya setuju bahwa nilai target sebenarnya adalah 0. Dari hasil perhitungan MSE, berdasarkan plot dan hasil prediksi data test, kita dapat menyimpulkan bahwa algoritma Random Forest merupakan algoritma yang memiliki hasil paling optimal diikuti oleh KNN dan terakhir adalah Boosting Algorithm.

Jadi, hasil analisis mendapatkan 5 variabel yang berpengaruh pada tingkat obesitas seseorang yaitu umuir, berat badan, riwayat keluarga penderita obesitas dan konsumsi makanan selain makanan utama. Model juga dapat meprediksi tingkat obesitas dengan sangat baik sehingga model ini dapat digunakan untuk memprediksi tingkat obesitas dari data yang baru.


## Daftar Pustaka
Palechor, Fabio Mendoza, dan Alexis de la Hoz Manotas. “_Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru, and Mexico._” Universidad de la Costa, CUC, Colombia, Vol. 25, 2019.

Widiantini, Winne dan Zarfiel Tafal. “Aktivitas Fisik, Stres, dan Obesitas pada Pegawai Negeri Sipil.” _Jurnal Kesehatan Masyarakat Nasional (Kesmas)_, Vol. 8, No. 7, 2014.