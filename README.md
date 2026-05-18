# Veri Bilimine Giriş: Makine Ağırlık Tahminleme Projesi

Bu proje, bir kurutma/çamaşır makinesinin tamburundaki yük ağırlığını (gram cinsinden) ölçmek için sensör verilerini kullanan bir makine öğrenmesi boru hattını (pipeline) içermektedir. `Ağırlık_Tahminlemesi.py` dosyası üzerinden çalışan bu sistem, cihazdan alınan elektriksel ve termal sinyalleri işleyerek yüksek doğruluklu ağırlık tahminleri üretir.

## 📌 Proje Açıklaması
Geleneksel ağırlık sensörleri (load cell vb.) yerine, makinenin çalışırken harcadığı güç, çektiği akım ve tambur içi sıcaklık değişimleri analiz edilerek "Sanal Sensör" (Virtual Sensor) mantığıyla ağırlık tahmini yapılması hedeflenmiştir. 
Proje, iki farklı veri kaynağından beslenir:
1. **DAQ (Veri Toplama Cihazı) Dosyaları:** Voltaj, akım, güç, enerji, cosfi ve debi (flowmeter) ölçümleri.
2. **MAGNUM Dosyaları:** T3/NTC sıcaklık sensörleri ve yazılımsal işlem döngüsü ölçümleri.

## 🛠 Kullanılan Yöntemler ve Veri İşleme

### 1. Veri Temizleme ve Ön İşleme
* **Dinamik Veri Eşleştirme:** Klasördeki DAQ ve MAGNUM dosyaları ağırlık ve koşu (Run) numaralarına göre otomatik olarak eşleştirilir.
* **Akıllı Veri Dönüşümü (`safe_float`):** Excel dosyalarından gelen ve farklı bölgesel formatlarda (virgüllü veya noktalı ondalık sayılar, örn: `229,89`) kaydedilmiş ham veriler, `NaN` (Sayı Değil) hatasına düşmeden güvenli bir şekilde sayısal formatlara dönüştürülür.
* **Gürültü Filtreleme & Segmentasyon:** Deneyin sadece stabil olduğu `250. saniye` ile `1500. saniye` arasındaki veri penceresi çekilerek, makinenin ilk ısınma veya son soğuma anlarındaki anormallikler dışarıda bırakılır.
* **Otomatik Veri Koruma:** Sensörlerin bozuk veya eksik kayıt aldığı deneyler (Örn: Voltajın kaydedilmediği dosyalar) tespit edilip modeli zehirlememesi için otomatik olarak **filtrenelek sistemden dışlanır**.

### 2. Özellik Mühendisliği (Feature Engineering)
Ham verilerden toplam 28 adet matematiksel ve fiziksel özellik çıkarılmıştır. Öne çıkan özellikler:
* **Normalize Edilmiş Güç (`p_norm`):** Voltaj dalgalanmalarından bağımsız, 230V'a göre düzeltilmiş ortalama güç.
* **Termal İndeks (`thermal_idx`):** T3 sensörünün ısınma eğimi (`t3_slope`) ile çekilen gücün birbirine oranı (Makinenin ısıl kapasitesinin ağırlığa olan direnci).
* **Fiziksel Etkileşimler:** `energy_per_flow` (debiyeye düşen enerji), `power_per_current` ve NTC ısıl oranları gibi türetilmiş güçlü metrikler.

### 3. Makine Öğrenmesi Modelleri
Ağırlık tahmini için iki güçlü ağaç tabanlı modelin **Ensemble (Topluluk)** mimarisi kullanılmıştır:
* **Gradient Boosting Regressor (%60 Ağırlık):** Hataları adım adım minimize etmeye odaklanan güçlü bir regresyon modeli.
* **Random Forest Regressor (%40 Ağırlık):** Aşırı öğrenmeyi (overfitting) engelleyen ve varyansı düşüren paralel orman modeli.
* **Hiperparametre Optimizasyonu:** `RandomizedSearchCV` ve `ParameterGrid` ile model hiperparametrelerinin dinamik olarak ayarlanabilmesine olanak sağlanmıştır.

### 4. Başarı Ölçümü ve Doğrulama
* **Leave-One-Out (LOO) Cross-Validation:** Veri setinin kısıtlı olduğu durumlarda modelin hiç görmediği bir deneyde ne kadar başarılı olacağını ölçmek için (Gerçek Genelleme Hatası) her bir test teker teker dışarıda bırakılarak eğitilip test edilmiştir.
* **Hata Metriği:** Ortalama Mutlak Yüzde Hata (MAPE - Mean Absolute Percentage Error) ve Gram (g) cinsinden mutlak sapmalar kullanılmıştır.

## 📊 Özet Sonuçlar
* **İşlenen Veri:** Klasördeki 68 adet deney (RUN 1, RUN 2, RUN 3) otomatik olarak algılanmış ve %100 başarıyla işlenmiştir. Bozuk sensör verileri başarıyla filtrelenmiştir.
* **Eğitim Performansı:** Eğitim setindeki tahminlerde ortalama hata **~%6 ila %11** aralığında seyretmiş, ortanca hata (Median Error) ise **%5-6** bandına kadar düşürülmüştür.
* **Görsel Raporlama:** Her çalışma sonrasında sonuçlar sadece terminale basılmakla kalmaz, aynı zamanda Matplotlib kullanılarak renk kodlu (±%2 tolerans sınırları) ve yüksek çözünürlüklü `.png` grafikleri olarak kaydedilir.

---
*Bu proje, makine sensör verilerinin ileri veri bilimi teknikleriyle anlamlandırılarak fiziksel sensör ihtiyacını ortadan kaldıran yazılımsal donanım (software sensor) uygulamalarına modern bir örnektir.*
