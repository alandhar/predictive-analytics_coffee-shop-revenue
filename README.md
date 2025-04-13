# Laporan Proyek Machine Learning Predictive Analytics - Muhamad Alan Dharma Saputro Setiawan

## Domain Proyek

Pertumbuhan industri kedai kopi di Indonesia semakin pesat, ditandai dengan menjamurnya berbagai jenis coffee shop, baik modern maupun tradisional. Di tengah kompetisi yang ketat ini, pengelolaan pendapatan harian menjadi tantangan utama bagi pelaku usaha kopi. Menurut penelitian Khoirotul Ummah dan Rahmat Agus Santoso (2023) dalam Revenue in Service Quality Perspective at A Traditional Coffee Shop in Balongpanggang Village, Gresik, keberhasilan kedai kopi tradisional dalam mempertahankan pendapatan dipengaruhi oleh kualitas layanan yang mencakup ketelitian, ketanggapan, hingga empati terhadap pelanggan. Sementara itu, studi oleh Budi Rahardjo et al. (2019) dalam Coffee Shop Business Model Analysis menekankan bahwa pengembangan strategi bisnis berbasis data dan inovasi model usaha sangat penting untuk menjaga keberlanjutan bisnis kedai kopi.

Berdasarkan kondisi tersebut, proyek ini bertujuan untuk membangun sistem predictive analytics yang mampu memprediksi pendapatan harian coffee shop menggunakan dataset operasional seperti jumlah pelanggan, nilai transaksi rata-rata, jam operasional, jumlah karyawan, pengeluaran promosi, dan lalu lintas lokasi. Permasalahan ini perlu diselesaikan karena pemilik coffee shop umumnya masih mengandalkan intuisi dalam pengambilan keputusan, padahal faktor-faktor operasional dapat diukur dan diprediksi secara kuantitatif. Dengan adanya model prediktif yang akurat, pelaku usaha dapat mengoptimalkan sumber daya, merespons tren dengan cepat, serta meningkatkan efisiensi dan daya saing bisnis secara berkelanjutan.

Referensi:
- [Revenue in Service Quality Perspective at A Traditional Coffee Shop in Balongpanggang Village, Gresik – Khoirotul Ummah & Rahmat Agus Santoso, 2023](https://journal.umg.ac.id/index.php/innovation/article/view/5358/3157)
- [Coffee Shop Business Model Analysis – Budi Rahardjo, Rokhani Hasbullah & Fahim M. Taqi, 2019](https://d1wqtxts1xzle7.cloudfront.net/81814341/pdf-libre.pdf?1646613357=&response-content-disposition=inline%3B+filename%3DCoffee_Shop_Business_Model_Analysis.pdf&Expires=1744383545&Signature=NgatOF4QR~AOOWvHeFPAAEIMdLZO7EdHJkkLHChmYdMbtGzWqvZ51AOYrg7rq9zVweGHklPwHsWbOhBwRGuohM4YN~dDf9F0Fa1TjrFII6jsfuFP28ZhB0wxmDnpgOCJY-mbCaRDZocbCTt3zWr1unaURvLvGfvKQt3DpA1-heie649EPom1V~gwjKhN-mkpIy~83W8m44BzcBeS0ZYfmRi3cvk9-Q-X0cdOGrtd9qvAWBtI9y2aaP-yr8KCJgWeaiRARvvu5YiOHyAiUqmvGkCa3iYGh0tX5xlkvcgY627l-nKc6gOeIlhm9UK1jPBUhpv6hWlHjngwIoCHPMspHQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

## Business Understanding

### Problem Statements

1. Prediksi pendapatan harian coffee shop menjadi tantangan karena pelaku usaha sering kali belum memiliki sistem analitik berbasis data yang akurat. Keputusan bisnis seperti stok harian, jumlah pegawai yang dijadwalkan, dan strategi promosi masih banyak didasarkan pada intuisi tanpa dasar kuantitatif yang kuat.
2. Tidak semua variabel operasional seperti jumlah pelanggan, pengeluaran pemasaran, atau jam operasional memiliki pengaruh yang sama terhadap pendapatan. Namun, pelaku usaha sering kesulitan dalam mengidentifikasi faktor mana yang paling signifikan, sehingga optimalisasi bisnis menjadi kurang efektif.
3. Minimnya sistem prediktif berbasis machine learning yang dapat diterapkan secara praktis dalam konteks usaha coffee shop menyebabkan keterbatasan dalam pengambilan keputusan strategis harian. Hal ini menghambat efisiensi operasional dan peluang peningkatan pendapatan secara berkelanjutan.

### Goals

1. Mengembangkan model prediktif berbasis machine learning untuk memperkirakan pendapatan harian coffee shop secara akurat berdasarkan variabel operasional seperti jumlah pelanggan, nilai pembelian rata-rata, jam operasional, dan pengeluaran promosi.
2. Mengidentifikasi fitur-fitur operasional paling berpengaruh terhadap pendapatan, sehingga pemilik usaha dapat fokus mengoptimalkan faktor yang paling berdampak secara finansial.
3. Menyediakan sistem prediksi yang praktis dan akurat, yang dapat digunakan untuk mendukung pengambilan keputusan harian berbasis data di lingkungan usaha coffee shop.

### Solution Statements

1. Menguji dan membandingkan beberapa algoritma regresi, yaitu:
    - Linear Regression sebagai model baseline.
    - Random Forest dan Gradient Boosting sebagai model ensemble non-linear.
    - Neural Network sebagai model deep learning.
2. Mengoptimalkan performa model Neural Network (deep learning) melalui peningkatan arsitektur jaringan, penambahan regularisasi (Dropout dan Batch Normalization), serta penggunaan EarlyStopping untuk mencegah overfitting dan memperoleh akurasi prediksi terbaik.
3. Melakukan feature engineering dan seleksi fitur berdasarkan analisis korelasi dan kontribusi terhadap target `Daily_Revenue`, untuk meningkatkan akurasi dan efisiensi model.
4. Menggunakan metrik evaluasi yang terukur, seperti:
    - Root Mean Squared Error (RMSE) untuk mengukur tingkat kesalahan prediksi dalam satuan dolar.
    - R² Score untuk menilai seberapa baik model menjelaskan variasi pendapatan harian.

## Data Understanding

Dataset yang digunakan dalam proyek ini berjudul "Coffee Shop Daily Revenue Prediction", yang diunduh dari Kaggle. Dataset ini berisi data operasional harian coffee shop sebanyak 2.000 entri dan 7 fitur numerik, serta satu kolom target yaitu `Daily_Revenue`. Data ini merepresentasikan kombinasi dari variabel internal seperti jumlah pelanggan dan karyawan, serta faktor eksternal seperti pengeluaran marketing dan lalu lintas lokasi.

Sumber Data : [Coffee Shop Daily Revenue Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/himelsarder/coffee-shop-daily-revenue-prediction-dataset/data)

---

### Variables in the Coffee Shop Daily Revenue:

- Number_of_Customers_Per_Day: Jumlah pelanggan yang datang setiap hari (50–499 pelanggan).
- Average_Order_Value: Nilai rata-rata pembelian per pelanggan dalam dolar ($2.50–$10.00).
- Operating_Hours_Per_Day: Total jam operasional coffee shop setiap hari (6–17 jam).
- Number_of_Employees: Jumlah karyawan yang bekerja per hari (2–14 orang).
- Marketing_Spend_Per_Day: Biaya pemasaran yang dikeluarkan setiap hari ($10.12–$499.74).
- Location_Foot_Traffic: Jumlah pejalan kaki yang melewati lokasi per jam (50–999 orang).
- Daily_Revenue (Target): Pendapatan harian dalam dolar, berkisar antara -58.95 hingga 5.114,60. Nilai negatif dihilangkan karena merupakan anomali yang tidak logis secara bisnis.

---

### Data Condition

- Jumlah Data: 2.000 baris × 7 kolom
- Missing Value: Tidak ditemukan nilai kosong pada kolom manapun (`df.isnull().sum()` = 0)
- Data Duplikat: Tidak ditemukan duplikasi (`df.duplicated().sum()` = 0)
- Outlier: Ditemukan satu outlier pada target `Daily_Revenue` dengan nilai negatif (-58.95) yang secara logis tidak valid, sehingga baris tersebut dihapus dalam tahap pembersihan data.

---

### Statistik Deskriptif

| Fitur                        | Mean     | Std Dev  | Min     | 25%     | 50%     | 75%     | Max      |
|-----------------------------|----------|----------|---------|---------|---------|---------|----------|
| Number_of_Customers_Per_Day | 274.30   | 129.44   | 50.00   | 164.00  | 275.00  | 386.00  | 499.00   |
| Average_Order_Value         | 6.26     | 2.18     | 2.50    | 4.41    | 6.30    | 8.12    | 10.00    |
| Operating_Hours_Per_Day     | 11.67    | 3.44     | 6.00    | 9.00    | 12.00   | 15.00   | 17.00    |
| Number_of_Employees         | 7.95     | 3.74     | 2.00    | 5.00    | 8.00    | 11.00   | 14.00    |
| Marketing_Spend_Per_Day     | 252.61   | 141.14   | 10.12   | 130.13  | 250.99  | 375.35  | 499.74   |
| Location_Foot_Traffic       | 534.89   | 271.66   | 50.00   | 302.00  | 540.00  | 767.00  | 999.00   |
| Daily_Revenue               | 1917.33  | 976.20   | -58.95  | 1140.08 | 1770.78 | 2530.46 | 5114.60  |

- Tidak ditemukan missing value maupun duplikasi.
- Outlier ditemukan pada `Daily_Revenue` (nilai negatif).
- Fitur-fitur menunjukkan skala berbeda, sehingga perlu dilakukan normalisasi (standardization) sebelum pelatihan model.

---

### Distribusi Fitur Numerik

![Distribusi Fitur](https://i.ibb.co.com/ymNbrVyt/image.png)

Visualisasi di atas menunjukkan distribusi dari setiap fitur numerik:
- Mayoritas fitur memiliki distribusi yang mendekati normal atau seragam.
- `Number_of_Customers_Per_Day`, `Operating_Hours_Per_Day`, dan `Average_Order_Value` terdistribusi relatif seimbang.
- Tidak terdapat skewness ekstrem yang memerlukan transformasi logaritmik.

---

### Distribusi Target `Daily_Revenue`

![Distribusi Target](https://i.ibb.co.com/WvzjqBZg/image.png)

Distribusi target `Daily_Revenue` menunjukkan:
- Kemencengan ke kanan (*right-skewed*), artinya sebagian besar pendapatan berada di kisaran menengah dengan beberapa hari berpendapatan sangat tinggi.
- Ditemukan 1 nilai outlier negatif, yaitu -58.95 yang tidak logis untuk konteks pendapatan, dan telah dibersihkan dari dataset.

---

### Korelasi Antar Variabel

![Korelasi Variabel](https://i.ibb.co.com/Z1YYzBBK/image.png)

Heatmap di atas menunjukkan korelasi antar variabel:
- Fitur dengan korelasi tertinggi terhadap target:
  - `Number_of_Customers_Per_Day` (r = 0.74)
  - `Average_Order_Value` (r = 0.54)
  - `Marketing_Spend_Per_Day` (r = 0.25)
- Fitur seperti `Location_Foot_Traffic`, `Number_of_Employees`, dan `Operating_Hours_Per_Day` memiliki korelasi lemah terhadap target.
---

## Data Preparation

### Drop Outlier
Pada tahap ini, data yang memiliki nilai `Daily_Revenue` negatif dihapus karena tidak logis secara bisnis. Pendapatan harian tidak mungkin bernilai negatif, sehingga data ini dianggap sebagai outlier dan dapat merusak kualitas pelatihan model.
```python
df = df[df['Daily_Revenue'] >= 0]
```

### Ferature Engineering

Fitur baru dibuat dari kombinasi fitur yang telah ada untuk meningkatkan informasi yang tersedia bagi model. Tujuan dari tahap ini adalah memperkenalkan variabel turunan yang lebih representatif terhadap kondisi operasional harian coffee shop.
```python
df_fe = df.copy()
df_fe['Customer_Spend'] = df_fe['Number_of_Customers_Per_Day'] * df_fe['Average_Order_Value']
df_fe['Revenue_per_Hour'] = df_fe['Daily_Revenue'] / df_fe['Operating_Hours_Per_Day']
df_fe['Customers_per_Hour'] = df_fe['Number_of_Customers_Per_Day'] / df_fe['Operating_Hours_Per_Day']
df_fe['Spend_per_Employee'] = df_fe['Marketing_Spend_Per_Day'] / df_fe['Number_of_Employees']
df_fe['Foot_Traffic_Efficiency'] = df_fe['Location_Foot_Traffic'] / df_fe['Number_of_Customers_Per_Day']
```
- **Customer_Spend**  
  Estimasi total pengeluaran pelanggan per hari. Dihitung dari jumlah pelanggan dikali rata-rata nilai pembelian. Sangat berkorelasi dengan pendapatan harian.
- **Revenue_per_Hour**  
  Rata-rata pendapatan yang diperoleh per jam operasional. Mengukur efisiensi waktu kerja.
- **Customers_per_Hour**  
  Rata-rata jumlah pelanggan per jam. Menggambarkan kepadatan pengunjung selama operasional.
- **Spend_per_Employee**  
  Biaya promosi yang dibagi per karyawan. Mengukur efisiensi alokasi biaya pemasaran terhadap tenaga kerja.
- **Foot_Traffic_Efficiency**  
  Rasio antara lalu lintas lokasi dan jumlah pelanggan. Menunjukkan efektivitas lokasi dalam menarik pelanggan dari keramaian.

Fitur-fitur ini dipilih berdasarkan potensi korelasi yang tinggi dan relevansi logis terhadap target `Daily_Revenue`.

### Feature Selection

Berdasarkan analisis korelasi terhadap target `Daily_Revenue`, dipilih lima fitur yang memiliki korelasi tinggi untuk digunakan dalam pelatihan model.

![Heatmap Korelasi terhadap Target](https://i.ibb.co.com/LzQ2sGDB/image.png)

Pemilihan ini bertujuan untuk menyederhanakan model tanpa mengurangi kualitas informasi yang dibutuhkan. Model diharapkan dapat mempelajari hubungan yang kuat dan relevan dengan target Daily_Revenue, sekaligus menghindari overfitting dan redundansi dari fitur yang kurang informatif. Adapun fitur yang dipilih antara lain:
- `Customer_Spend`
- `Revenue_per_Hour`
- `Number_of_Customers_Per_Day`
- `Customers_per_Hour`
- `Average_Order_Value`

### Feature Scaling

Model seperti Neural Network dan SVR sensitif terhadap skala nilai fitur. Oleh karena itu, dilakukan standarisasi data menggunakan `StandardScaler` agar setiap fitur memiliki rata-rata 0 dan standar deviasi 1.
```python
X = df_fe[selected_features]
y = df_fe['Daily_Revenue']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Train-Test Split

Dataset kemudian dibagi menjadi dua bagian:
- **Training set (80%)**: digunakan untuk melatih model.
- **Testing set (20%)**: digunakan untuk mengevaluasi performa model.

Pembagian dilakukan secara acak tetapi konsisten dengan `random_state=42`.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

## Modeling

Tahapan ini bertujuan untuk membangun dan mengevaluasi berbagai model machine learning dalam memprediksi `Daily_Revenue` berdasarkan fitur-fitur terpilih yang telah diproses pada tahap sebelumnya. Beberapa algoritma regresi digunakan dan dibandingkan performanya untuk menentukan model terbaik.

### 1. Linear Regression

Linear Regression adalah model regresi paling dasar yang mengasumsikan hubungan linear antara variabel input dan target. Model ini sangat cocok digunakan sebagai baseline karena cepat, efisien, dan mudah dipahami, serta digunakan dengan parameter default dari `sklearn.LinearRegression()`. Namun, model ini memiliki keterbatasan dalam menangkap pola non-linear dan sangat sensitif terhadap outlier.

---

### 2. Random Forest Regressor

Random Forest adalah model ensemble berbasis decision tree yang bekerja dengan membangun banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan stabilitas prediksi. Model ini sangat efektif dalam menangani data non-linear dan robust terhadap outlier, serta dilatih dengan menetapkan parameter `random_state=42` untuk memastikan hasil yang konsisten. Meskipun demikian, model ini kurang interpretatif dan dapat memerlukan lebih banyak sumber daya memori pada dataset besar.

---

### 3. Gradient Boosting Regressor

Gradient Boosting adalah model boosting yang membangun serangkaian model secara iteratif, di mana setiap model baru berusaha memperbaiki kesalahan dari model sebelumnya. Model ini sangat presisi dalam menangani pola data yang kompleks dan dilatih menggunakan parameter `random_state=42` untuk memastikan reprodusibilitas. Meskipun akurat, model ini memerlukan waktu pelatihan yang lebih lama dan rentan terhadap overfitting jika tidak dilakukan tuning secara tepat.

---

### 4. Neural Network (Deep Learning)

Neural Network menawarkan fleksibilitas tertinggi karena dapat membentuk arsitektur yang kompleks sesuai kebutuhan. Dalam proyek ini, NN menunjukkan performa paling baik dengan RMSE paling rendah dan R² tertinggi. Model ini juga menunjukkan kestabilan saat training berkat penambahan dropout, batch normalization, dan early stopping.

- **Arsitektur Model**:
  - Input Layer (jumlah fitur)
  - Dense(128) → ReLU → BatchNormalization → Dropout(0.3)
  - Dense(64) → ReLU → BatchNormalization → Dropout(0.3)
  - Dense(32) → ReLU
  - Output Layer: Dense(1)
- **Compile**:
  - Optimizer: `Adam(lr=0.001)`
  - Loss: `Mean Squared Error`
- **Training**:
  - Epoch: 200
  - Batch size: 32
  - Validation split: 10%
  - Callbacks: `EarlyStopping(patience=10, restore_best_weights=True)`

Model Neural Network dipilih sebagai model terbaik dalam proyek ini karena memberikan performa prediksi yang paling optimal setelah dilakukan proses tuning. Penyesuaian arsitektur model dilakukan dengan menambahkan lapisan **Dropout** dan **BatchNormalization**, serta penerapan teknik **EarlyStopping** untuk mencegah overfitting.

![Training vs Validation Loss](https://i.ibb.co.com/4nR8TFRL/image.png)

Model Neural Network sangat cocok digunakan karena:

- Data bersifat **numerik kontinu** dan memiliki **interaksi kompleks antar fitur**.
- Fitur-fitur hasil rekayasa (*feature engineering*) membentuk hubungan non-linear, yang dapat lebih efektif ditangkap oleh model Neural Network dibanding model linier atau tree-based.
- Model dapat dioptimalkan lebih lanjut dengan fleksibilitas tinggi dalam hal struktur dan strategi pelatihan.

---

### Kelebihan & Kekurangan Model
| Model               | Kelebihan                                                                 | Kekurangan                                                                 |
|---------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Linear Regression** | Sederhana dan cepat dilatih; mudah diinterpretasi                     | Tidak mampu menangani hubungan non-linear; rentan terhadap outlier      |
| **Random Forest**     | Menangani non-linearitas; robust terhadap outlier dan noise           | Kurang interpretatif; konsumsi memori tinggi pada dataset besar         |
| **Gradient Boosting** | Akurasi tinggi; dapat menyesuaikan terhadap outlier dan error         | Lebih lambat dibanding Random Forest; rentan overfitting tanpa tuning   |
| **Neural Network**    | Sangat fleksibel dan powerful; dapat dikustomisasi penuh              | Butuh waktu dan sumber daya lebih besar; sensitif terhadap skala data dan overfitting |

## Evaluation

Tahapan evaluasi dilakukan untuk menilai seberapa baik model memprediksi nilai `Daily_Revenue` dari data coffee shop yang diberikan. Karena ini merupakan kasus **regresi**, maka digunakan dua metrik evaluasi utama, yaitu:

### 1. Root Mean Squared Error (RMSE)

RMSE mengukur rata-rata kesalahan prediksi antara nilai aktual dan nilai prediksi model, dalam satuan yang sama dengan target (USD). Semakin kecil nilai RMSE, semakin baik performa model.

### 2. R² Score (Koefisien Determinasi)

R² mengukur seberapa baik fitur-fitur input menjelaskan variabilitas dari target (`Daily_Revenue`). Nilai R² berkisar dari 0 hingga 1, di mana nilai mendekati 1 menunjukkan bahwa model menjelaskan sebagian besar variansi target.

---

Berikut adalah hasil evaluasi model pada data uji:

| Model              | RMSE (Train) | RMSE (Test) | R² Score (Test) |
|--------------------|--------------|-------------|------------------|
| **Neural Network** | 75.30        | **69.34**   | **0.995**        |
| Linear Regression  | 183.16       | 172.16      | 0.969            |
| Gradient Boosting  | 142.93       | 190.52      | 0.962            |
| Random Forest      | 74.88        | 202.91      | 0.957            |

- **Neural Network** menunjukkan performa terbaik dengan **RMSE Test paling kecil** (69.34) dan **R² Score tertinggi** (0.995), menandakan bahwa model ini sangat akurat dan mampu menjelaskan hampir seluruh variabilitas pendapatan harian.
- **Linear Regression** cukup baik namun tidak cukup kuat dalam menangkap hubungan non-linear pada data.
- **Gradient Boosting** dan **Random Forest** menunjukkan performa tinggi di training set namun tidak sebaik Neural Network di data uji, menandakan kemungkinan overfitting ringan.
- Selisih nilai RMSE antara training dan test pada Neural Network juga kecil, yang berarti model memiliki **kemampuan generalisasi yang sangat baik**.

---

Grafik berikut menunjukkan perbandingan RMSE antara data training dan test untuk semua model:

![Perbandingan RMSE Train vs Test](https://i.ibb.co.com/Fkfj8t7p/image.png)

Model Neural Network yang telah dioptimasi menunjukkan performa terbaik dalam memprediksi `Daily_Revenue`. Dengan akurasi tinggi dan stabilitas yang konsisten di data uji, model ini layak dipilih sebagai solusi akhir dari proyek predictive analytics ini.


