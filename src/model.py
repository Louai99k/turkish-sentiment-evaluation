"""
Türkçe Duygu Analizi için BERTurk modeli
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from src.protocol_constants import BERTURK_MODEL_ID, BERTURK_MODEL_REVISION


class BERTurkSentimentClassifier(nn.Module):
    """
    BERTurk tabanlı duygu sınıflandırıcı
    """
    
    def __init__(
        self,
        model_name=BERTURK_MODEL_ID,
        model_revision=BERTURK_MODEL_REVISION,
        num_classes=3,
        dropout=0.3,
        local_files_only=False,
    ):
        super(BERTurkSentimentClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.model_revision = model_revision
        
        # Önceden eğitilmiş BERTurk'ü yükle
        self.bert = AutoModel.from_pretrained(
            model_name,
            revision=model_revision,
            local_files_only=local_files_only,
        )
        
        # BERT yapılandırmasından gizli katman boyutunu al
        self.hidden_size = self.bert.config.hidden_size
        
        # Sınıflandırma katmanı
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        İleri geçiş
        
        Args:
            input_ids: Token ID'leri (batch_size, seq_length)
            attention_mask: Dikkat maskesi (batch_size, seq_length)
        
        Returns:
            logits: Sınıflandırma logitleri (batch_size, num_classes)
        """
        # BERT çıktılarını al
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token temsilini kullan (ilk token)
        pooled_output = outputs.pooler_output
        
        # Dropout ve sınıflandırma katmanını uygula
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def freeze_bert_layers(self, num_layers_to_freeze=10):
        """
        Eğitim süresini azaltmak için BERT'in ilk N katmanını dondur
        
        Args:
            num_layers_to_freeze: Dondurulacak BERT katman sayısı (0-12)
        """
        # Gömmeleri dondur
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # İlk N kodlayıcı katmanını dondur
        for i in range(num_layers_to_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"İlk {num_layers_to_freeze} BERT katmanı donduruldu")
    
    def unfreeze_all(self):
        """Tüm parametreleri çöz"""
        for param in self.parameters():
            param.requires_grad = True
        print("Tüm katmanlar çözüldü")


def get_model(
    num_classes=3,
    model_name=BERTURK_MODEL_ID,
    model_revision=BERTURK_MODEL_REVISION,
    dropout=0.3,
    device='cuda',
    local_files_only=False,
):
    """
    Modeli oluştur ve başlat
    
    Args:
        num_classes: Çıktı sınıflarının sayısı
        model_name: Önceden eğitilmiş modelin adı
        device: Modelin yükleneceği cihaz
    
    Returns:
        model: Başlatılmış model
    """
    model = BERTurkSentimentClassifier(
        model_name=model_name,
        model_revision=model_revision,
        num_classes=num_classes,
        dropout=dropout,
        local_files_only=local_files_only,
    )
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Model testi
    print("Model test ediliyor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    model = get_model(num_classes=3, device=device)
    
    # Sahte girdi oluştur
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    # İleri geçiş
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"\nGirdi boyutu: {input_ids.shape}")
    print(f"Çıktı boyutu: {logits.shape}")
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Eğitilebilir parametreler: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\nModel hazır!")
