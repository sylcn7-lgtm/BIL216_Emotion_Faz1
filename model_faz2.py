import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def model_faz2_calistir():
    csv_yolu = r"C:\Users\aixwa\Desktop\BIL216_Faz1\Dataset\extracted_emotion_features_faz2.csv"
    
    if not os.path.exists(csv_yolu):
        print("❌ HATA: Öznitelik CSV dosyası bulunamadı! Önce ilk kodu çalıştır.")
        return
        
    df = pd.read_csv(csv_yolu)
    
    # Modelleme için X ve y ayır
    X = df.drop(columns=['Dosya_Adı', 'Gercek_Duygu'])
    y = df['Gercek_Duygu']
    
    # %80 Eğitim, %20 Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Faz 2 Gelişmiş Random Forest Konfigürasyonu
    model = RandomForestClassifier(
        n_estimators=250,       # Ağaç sayısı artırıldı
        max_depth=18,           # Karar derinliği optimize edildi
        min_samples_split=4,    # Dallanma kontrolü eklendi
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Tahmin ve Skor Hesaplama
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    
    print("\n" + "="*50)
    print(f"🚀 FAZ 2 GERÇEK VERİ SETİ BAŞARI ORANI: %{acc:.2f}")
    print("="*50)
    print("\n📊 DETAYLI SINIFLANDIRMA RAPORU:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    model_faz2_calistir()