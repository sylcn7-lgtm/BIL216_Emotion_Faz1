import os, librosa, pandas as pd, numpy as np

def final_operasyon():
    # Senin verdiğin tam yol
    yol = r"C:\Users\aixwa\Desktop\BIL216_Faz1\Dataset"
    
    # Klasördeki TÜM dosyaları al (Sadece .csv ve .xlsx olmayanları işle)
    dosyalar = [f for f in os.listdir(yol) if not f.endswith(('.csv', '.xlsx'))]
    
    print(f"--- TARAMA TAMAMLANDI ---")
    print(f"Klasörde işlenecek {len(dosyalar)} adet dosya bulundu.")
    
    veriler = []
    for isim in dosyalar:
        tam_yol = os.path.join(yol, isim)
        print(f"İşleniyor: {isim}")
        
        duygu = "Notr"
        if "Mutlu" in isim: duygu = "Mutlu"
        elif "Ofkeli" in isim: duygu = "Ofkeli"
        elif "Saskin" in isim: duygu = "Saskin"
        elif "Uzgun" in isim: duygu = "Uzgun"
        
        try:
            # Uzantıya bakmadan sesi yüklemeyi dene
            y, sr = librosa.load(tam_yol, sr=22050)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
            d = {"Dosya_Adı": isim, "Gercek_Duygu": duygu}
            for i in range(13): d[f"MFCC_{i+1}"] = mfcc[i]
            veriler.append(d)
        except:
            continue
            
    if veriler:
        df = pd.DataFrame(veriler)
        csv_yolu = os.path.join(yol, "extracted_emotion_features.csv")
        df.to_csv(csv_yolu, index=False)
        print(f"\n✅ BAŞARILI! {len(veriler)} dosya işlendi.")

if __name__ == "__main__":
    final_operasyon()