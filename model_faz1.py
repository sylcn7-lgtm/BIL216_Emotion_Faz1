import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def final_skor():
    # Dosyayı oku
    df = pd.read_csv("Dataset/extracted_emotion_features.csv")
    
    # X ve y ayır (Dosya adı ve duygu sütunlarını özelliklerden çıkar)
    X = df.drop(columns=['Dosya_Adı', 'Gercek_Duygu'])
    y = df['Gercek_Duygu']
    
    # Train-test ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modeli eğit
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Sonucu yazdır
    y_pred = model.predict(X_test)
    print(f"\n🎉 FAZ 1 BAŞARI ORANI: %{accuracy_score(y_test, y_pred) * 100:.2f}")

if __name__ == "__main__":
    final_skor()