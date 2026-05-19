import os
import pandas as pd
import numpy as np
import librosa

def gurultu_ekle(y, gurultu_orani=0.005):
    # Beyaz gürültü ekleme
    noise = np.random.randn(len(y))
    return y + gurultu_orani * noise

def hiz_degistir(y, hiz=1.15):
    # Sesin hızını değiştirme (Time Stretch)
    return librosa.effects.time_stretch(y, rate=hiz)

def ton_kaydir(y, sr, bas_basamak=1.5):
    # Sesin frekans tonunu kaydırma (Pitch Shift)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=bas_basamak)

def oznitelik_cikari_faz2():
    klasor_yolu = r"C:\Users\aixwa\Desktop\BIL216_Faz1\Dataset"
    
    uzantilar = (".wav", ".aac", ".ogg")
    ses_dosyalari = [f for f in os.listdir(klasor_yolu) if f.lower().endswith(uzantilar)]
    
    if not ses_dosyalari:
        print("❌ HATA: Dataset klasöründe işlenecek ses dosyası bulunamadı!")
        return
        
    print(f"📁 Klasörde {len(ses_dosyalari)} adet orijinal dosya bulundu.")
    print("⏳ Gerçek zamanlı Data Augmentation ve Öznitelik Çıkarımı başlıyor...")
    
    veriler = []
    duygu_sozlugu = {"mutlu": "Mutlu", "notr": "Notr", "ofkeli": "Ofkeli", "saskin": "Saskin", "uzgun": "Uzgun"}

    for index, dosya in enumerate(ses_dosyalari, 1):
        tam_yol = os.path.join(klasor_yolu, dosya)
        
        # Dosya adından duygu etiketini yakala
        gercek_duygu = "Bilinmiyor"
        for anahtar, deger in duygu_sozlugu.items():
            if anahtar in dosya.lower():
                gercek_duygu = deger
                break
                
        try:
            # Orijinal sesi yükle
            y, sr = librosa.load(tam_yol, sr=22050)
            
            # Varyasyonları oluştur (Orijinal + 3 Sentetik Varyasyon)
            varyasyonlar = [
                ("Orijinal", y),
                ("Gurultulu", gurultu_ekle(y)),
                ("Hizli", hiz_degistir(y, hiz=1.15)),
                ("TonKaymis", ton_kaydir(y, sr, bas_basamak=1.5))
            ]
            
            for var_adi, var_sinyal in varyasyonlar:
                # 13 Katsayılı MFCC
                mfccs = librosa.feature.mfcc(y=var_sinyal, sr=sr, n_mfcc=13)
                mfcc_ortalamalar = np.mean(mfccs, axis=1)
                
                # Sıfır Geçiş Oranı (ZCR)
                zcr = librosa.feature.zero_crossing_rate(y=var_sinyal)
                zcr_ortalamasi = np.mean(zcr)
                
                # RMS Enerji
                rms = librosa.feature.rms(y=var_sinyal)
                rms_ortalamasi = np.mean(rms)
                
                satir = {
                    "Dosya_Adı": f"{var_adi}_{dosya}",
                    "Gercek_Duygu": gercek_duygu,
                    "ZCR": zcr_ortalamasi,
                    "RMS_Enerji": rms_ortalamasi
                }
                for idx, deger in enumerate(mfcc_ortalamalar):
                    satir[f"MFCC_{idx+1}"] = deger
                    
                veriler.append(satir)
                
            if index % 20 == 0 or index == len(ses_dosyalari):
                print(f"▓ [{index}/{len(ses_dosyalari)}] dosya başarıyla işlendi...")
            
        except Exception as e:
            pass
            
    df = pd.DataFrame(veriler)
    csv_yolu = os.path.join(klasor_yolu, "extracted_emotion_features_faz2.csv")
    df.to_csv(csv_yolu, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 ÖZNİTELİK ÇIKARIMI BİTTİ!")
    print(f"📊 Toplam Genişletilmiş Veri Satırı: {len(df)}")
    print(f"💾 Dosya şuraya kaydedildi: {csv_yolu}")
    print("="*50)

if __name__ == "__main__":
    oznitelik_cikari_faz2()