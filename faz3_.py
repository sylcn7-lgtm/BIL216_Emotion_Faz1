import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def extract_features_faz3(y, sr):
    features = []

    # 1. ZCR (Mean & STD)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 2. RMS Enerji (Mean & STD)
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 3. Spectral Centroid (Mean & STD)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    features.append(np.std(spec_cent))

    # 4. Gelişmiş MFCC, Delta ve Delta-Delta Analizi
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    for m, d, d2 in zip(mfcc, delta_mfcc, delta2_mfcc):
        features.append(np.mean(m))
        features.append(np.std(m))
        features.append(np.mean(d))
        features.append(np.std(d))
        features.append(np.mean(d2))
        features.append(np.std(d2))

    return features

# --- GRUP VE DUYGU AYARLARI ---
dataset_path = "Dataset"
X = []
y = []

duygu_sozlugu = {"mutlu": "happy", "notr": "neutral", "ofkeli": "angry", "saskin": "surprise", "uzgun": "sad"}

print("🚀 Faz 3 OPTİMİZE: Veri artırımı (Augmentation) uygulanarak öznitelikler çıkarılıyor...")

if not os.path.exists(dataset_path):
    print(f"❌ HATA: '{dataset_path}' klasörü bulunamadı!")
else:
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.lower().endswith((".wav", ".aac", ".ogg")):
                file_path = os.path.join(root, file_name)
                
                gercek_duygu = None
                for anahtar, deger in duygu_sozlugu.items():
                    if anahtar in file_name.lower():
                        gercek_duygu = deger
                        break
                
                if not gercek_duygu:
                    for duygu_ing in duygu_sozlugu.values():
                        if duygu_ing in file_name.lower():
                            gercek_duygu = duygu_ing
                            break
                
                if gercek_duygu:
                    try:
                        # Orijinal Sesi Yükle
                        y_audio, sr_audio = librosa.load(file_path, duration=3, offset=0.5)
                        
                        # 1. Orijinal Öznitelikleri Çıkar
                        feat_orig = extract_features_faz3(y_audio, sr_audio)
                        X.append(feat_orig)
                        y.append(gercek_duygu)
                        
                        # 2. Veri Artırımı: Sesi %15 Hızlandır (Model daha çok varyasyon görsün)
                        y_fast = librosa.effects.time_stretch(y_audio, rate=1.15)
                        feat_fast = extract_features_faz3(y_fast, sr_audio)
                        X.append(feat_fast)
                        y.append(gercek_duygu)
                        
                    except Exception as e:
                        pass

print(f"📊 Veri Artırımı Sonrası Toplam Örnek Sayısı: {len(X)}")

if len(X) == 0:
    print("❌ HATA: Veri okunamadı.")
else:
    columns = ["ZCR_Mean", "ZCR_Std", "RMS_Mean", "RMS_Std", "Spectral_Centroid_Mean", "Spectral_Centroid_Std"]
    for i in range(13):
        columns.extend([f"MFCC_{i+1}_Mean", f"MFCC_{i+1}_Std", f"Delta_{i+1}_Mean", f"Delta_{i+1}_Std", f"Delta2_{i+1}_Mean", f"Delta2_{i+1}_Std"])

    df = pd.DataFrame(X, columns=columns)
    df["Emotion"] = y
    df.to_csv("faz3_advanced_features.csv", index=False)

    X_data = df.drop(columns=["Emotion"])
    y_data = df["Emotion"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.2, random_state=42)

    # Maksimum Güçte XGBoost Konfigürasyonu
    model = XGBClassifier(
        n_estimators=500,         # Ağaç sayısı artırıldı
        max_depth=7,              # Derinlik artırıldı
        learning_rate=0.03,       # Daha hassas adımlarla öğrenme
        subsample=0.85,           # Veri örnekleme oranı optimize edildi
        colsample_bytree=0.85,
        random_state=42,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*45)
    print(f"🏆 FAZ 3 ULTRA PERFORMANS ACCURACY: %{accuracy*100:.2f}")
    print("="*45 + "\n")

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek Duygu")
    plt.title("Faz 3 - Confusion Matrix (XGBoost - Ultra)")
    plt.savefig("results/confusion_matrix_faz3.png")