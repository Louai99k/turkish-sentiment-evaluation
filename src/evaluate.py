"""
Duygu analizi modeli için değerlendirme araçları
"""
import torch
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model, data_loader, device, label_mapping=None):
    """
    Bir veri seti üzerinde modeli değerlend
    
    Args:
        model: Değerlendirilecek model
        data_loader: Veri seti için DataLoader
        device: Değerlendirmenin çalıştırılacağı cihaz
        label_mapping: Etiket ID'lerini isimlere eşleyen sözlük
    
    Returns:
        Değerlendirme metrikleriyle sözlük
    """
    if label_mapping is None:
        label_mapping = {0: 'Positive', 1: 'Negative', 2: 'Notr'}
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # İleri geçiş
            logits = model(input_ids, attention_mask)
            
            # Tahminleri al
            predictions = torch.argmax(logits, dim=1)
            
            # Sonuçları sakla
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    # Numpy dizilerine dönüştür
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # Karışıklık matrisi
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Sonuçları yazdır
    print("\n" + "="*50)
    print("DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    print(f"Doğruluk: {accuracy:.4f}")
    print(f"Ağırlıklı F1 Skoru: {f1_weighted:.4f}")
    print("\nSınıf bazında metrikler:")
    for i, label_name in sorted(label_mapping.items()):
        print(f"  {label_name}:")
        print(f"    Kesinlik: {precision[i]:.4f}")
        print(f"    Duyarlılık: {recall[i]:.4f}")
        print(f"    F1 Skoru: {f1_per_class[i]:.4f}")
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(
        all_labels, 
        all_predictions,
        target_names=[label_mapping[i] for i in sorted(label_mapping.keys())]
    ))
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits
    }


def plot_confusion_matrix(cm, label_mapping=None, save_path='confusion_matrix.png'):
    """
    Karışıklık matrisini çiz
    
    Args:
        cm: Karışıklık matrisi
        label_mapping: Etiket ID'lerini isimlere eşleyen sözlük
        save_path: Grafiğin kaydedileceği yol
    """
    if label_mapping is None:
        label_mapping = {0: 'Positive', 1: 'Negative', 2: 'Notr'}
    
    plt.figure(figsize=(10, 8))
    
    # Yüzdeleri hesapla
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Hem sayı hem yüzde ile açıklamalar oluştur
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(
        cm, 
        annot=annot, 
        fmt='', 
        cmap='Blues',
        xticklabels=[label_mapping[i] for i in sorted(label_mapping.keys())],
        yticklabels=[label_mapping[i] for i in sorted(label_mapping.keys())],
        cbar_kws={'label': 'Sayı'}
    )
    
    plt.title('Karışıklık Matrisi', fontsize=16, fontweight='bold')
    plt.ylabel('Gerçek Etiket', fontsize=12)
    plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nKarışıklık matrisi kaydedildi: {save_path}")
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """
    Eğitim geçmişini çiz
    
    Args:
        history: Eğitim ve doğrulama kayıpları/metrikleri içeren sözlük
        save_path: Grafiğin kaydedileceği yol
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Kayıp grafiği
    axes[0].plot(history['train_loss'], label='Eğitim Kaybı', marker='o')
    axes[0].plot(history['val_loss'], label='Doğrulama Kaybı', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Kayıp')
    axes[0].set_title('Eğitim ve Doğrulama Kaybı')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 skorunu çiz
    if 'val_f1' in history:
        axes[1].plot(history['val_f1'], label='Doğrulama F1', marker='s', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Skoru')
        axes[1].set_title('Doğrulama F1 Skoru')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEğitim geçmişi kaydedildi: {save_path}")
    plt.close()


def analyze_errors(model, data_loader, tokenizer, device, num_examples=10, label_mapping=None):
    """
    Yanlış sınıflandırılmış örnekleri analiz et
    
    Args:
        model: Model
        data_loader: DataLoader
        tokenizer: Metinleri çözmek için tokenizer
        device: Cihaz
        num_examples: Gösterilecek örnek sayısı
        label_mapping: Etiket eşleme
    """
    if label_mapping is None:
        label_mapping = {0: 'Positive', 1: 'Negative', 2: 'Notr'}
    
    model.eval()
    
    errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            # Yanlış sınıflandırılmış örnekleri bul
            mask = predictions != labels
            
            for i in range(len(predictions)):
                if mask[i]:
                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    errors.append({
                        'text': text,
                        'true_label': label_mapping[labels[i].item()],
                        'predicted_label': label_mapping[predictions[i].item()],
                        'logits': logits[i].cpu().numpy()
                    })
                    
                    if len(errors) >= num_examples:
                        break
            
            if len(errors) >= num_examples:
                break
    
    print("\n" + "="*50)
    print("YANLIŞ SINIFLANDIRILMIŞ ÖRNEKLER")
    print("="*50)
    
    for i, error in enumerate(errors, 1):
        print(f"\nÖrnek {i}:")
        print(f"Metin: {error['text'][:200]}...")
        print(f"Gerçek Etiket: {error['true_label']}")
        print(f"Tahmin Edilen Etiket: {error['predicted_label']}")
        print(f"Logitler: {error['logits']}")
        print("-" * 50)


if __name__ == "__main__":
    print("Değerlendirme araçları modülü hazır!")
