"""
TF-IDF + Lojistik Regresyon kullanan Baseline modeli
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


class BaselineModel:
    """TF-IDF + Lojistik Regresyon baseline"""
    
    def __init__(self, max_features=10000, class_weight='balanced'):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigram ve bigram
            min_df=2
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=42
        )
        self.label_mapping = {0: 'Positive', 1: 'Negative', 2: 'Notr'}
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """
        Baseline modeli eğit
        
        Args:
            train_texts: Eğitim metinleri
            train_labels: Eğitim etiketleri
            val_texts: Doğrulama metinleri (opsiyonel)
            val_labels: Doğrulama etiketleri (opsiyonel)
        """
        print("Baseline model eğitiliyor...")
        
        # TF-IDF vektörleştiriciyi uyarla
        X_train = self.vectorizer.fit_transform(train_texts)
        
        # Sınıflandırıcıyı eğit
        self.classifier.fit(X_train, train_labels)
        
        # Eğitim seti üzerinde değerlendirme
        train_pred = self.classifier.predict(X_train)
        train_f1 = f1_score(train_labels, train_pred, average='weighted')
        print(f"Eğitim F1 Skoru: {train_f1:.4f}")
        
        # Sağlandıysa doğrulama seti üzerinde değerlendirme
        if val_texts is not None and val_labels is not None:
            X_val = self.vectorizer.transform(val_texts)
            val_pred = self.classifier.predict(X_val)
            val_f1 = f1_score(val_labels, val_pred, average='weighted')
            print(f"Doğrulama F1 Skoru: {val_f1:.4f}")
    
    def evaluate(self, test_texts, test_labels):
        """
        Modeli değerlendirme
        
        Args:
            test_texts: Test metinleri
            test_labels: Test etiketleri
        
        Returns:
            Değerlendirme metrikleriyle sözlük
        """
        # Test metinlerini dönüştür
        X_test = self.vectorizer.transform(test_texts)
        
        # Tahmin et
        predictions = self.classifier.predict(X_test)
        
        # Metrikleri hesapla
        f1 = f1_score(test_labels, predictions, average='weighted')
        
        # Sınıflandırma raporunu yazdır
        print("\nSınıflandırma Raporu:")
        print(classification_report(
            test_labels, 
            predictions,
            target_names=[self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        ))
        
        # Karışıklık matrisi
        cm = confusion_matrix(test_labels, predictions)
        
        return {
            'f1_score': f1,
            'predictions': predictions,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix_baseline.png'):
        """Karışıklık matrisini çiz"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.label_mapping[i] for i in sorted(self.label_mapping.keys())],
            yticklabels=[self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        )
        plt.title('Karışıklık Matrisi - Baseline Model')
        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nKarışıklık matrisi kaydedildi: {save_path}")
        plt.close()
    
    def save(self, filepath='models/baseline_model.pkl'):
        """Modeli kaydet"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'classifier': self.classifier}, f)
        print(f"Model kaydedildi: {filepath}")
    
    def load(self, filepath='models/baseline_model.pkl'):
        """Modeli yükle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
        print(f"Model yüklendi: {filepath}")


if __name__ == "__main__":
    # Örnek kullanım
    print("Baseline model test ediliyor...")
    
    # Normalde gerçek veri seti kullanılır
    from data_loader import load_winvoker_dataset, preprocess_dataset
    
    # Veri setini yükle
    dataset = load_winvoker_dataset()
    data_splits = preprocess_dataset(dataset, test_size=0.2, val_size=0.1)
    
    # Baseline modeli oluştur
    model = BaselineModel()
    
    # Eğit
    model.train(
        train_texts=data_splits['train']['text'].values,
        train_labels=data_splits['train']['label'].values,
        val_texts=data_splits['val']['text'].values,
        val_labels=data_splits['val']['label'].values
    )
    
    # Değerlendirme
    results = model.evaluate(
        test_texts=data_splits['test']['text'].values,
        test_labels=data_splits['test']['label'].values
    )
    
    print(f"\nTest F1 Skoru: {results['f1_score']:.4f}")
    
    # Karışıklık matrisini çiz
    model.plot_confusion_matrix(results['confusion_matrix'])
    
    # Modeli kaydet
    model.save()
    
    print("\nBaseline model hazır!")
