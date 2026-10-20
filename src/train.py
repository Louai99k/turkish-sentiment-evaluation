"""
BERTurk Duygu Analizi için Eğitim Scripti
"""
import json
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
from src.focal_loss import MulticlassFocalLoss


class Trainer:
    """BERTurk modeli için Eğitim sınıfı"""
    
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        class_weights,
        device,
        learning_rate=2e-5,
        epochs=3,
        warmup_steps=0,
        output_dir='models',
        loss_mode='weighted',
        history_path=None,
        resume_from=None,
        checkpoint_metadata=None,
        checkpoint_schema_version='berturk-checkpoint-v2.0',
        focal_gamma=None,
        focal_alpha='none',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.output_dir = output_dir
        self.loss_mode = loss_mode
        self.history_path = Path(history_path) if history_path else None
        self.resume_checkpoint_path = Path(output_dir) / "resume_checkpoint.pt"
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.checkpoint_schema_version = checkpoint_schema_version
        
        # Keep the historical CE paths exactly as they were for continuity.
        if loss_mode == 'weighted':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        elif loss_mode == 'unweighted':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_mode == 'focal':
            if focal_gamma is None:
                raise ValueError('Focal loss requires an explicit gamma')
            if focal_alpha not in {'none', 'balanced'}:
                raise ValueError(f'Unsupported focal_alpha: {focal_alpha}')
            alpha = class_weights.to(device) if focal_alpha == 'balanced' else None
            self.criterion = MulticlassFocalLoss(gamma=focal_gamma, alpha=alpha)
        else:
            raise ValueError(f"Desteklenmeyen loss_mode: {loss_mode}")
        
        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        
        # Öğrenme oranı zamanlayıcı
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Geçmiş
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        # En iyi model takibi
        self.best_val_loss = float('inf')
        self.best_val_f1 = -float('inf')
        self.best_model_path = None
        self.start_epoch = 0
        self.patience_counter = 0
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)

        if resume_from is not None:
            self.load_resume_checkpoint(resume_from)

    @staticmethod
    def _atomic_torch_save(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def _write_history(self):
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_name(self.history_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.history_path)

    def _rng_state(self):
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        loader_generator = getattr(self.train_loader, 'generator', None)
        if loader_generator is not None:
            state['train_loader_generator'] = loader_generator.get_state()
        return state

    def _restore_rng_state(self, state):
        if not state:
            return
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        # RNG setters consume CPU ByteTensors.  Keep this normalization
        # defensive so checkpoints remain resumable even when a caller loaded
        # their tensor payloads through a CUDA map_location.
        torch.set_rng_state(state['torch_cpu'].detach().cpu().to(dtype=torch.uint8))
        if torch.cuda.is_available() and 'torch_cuda' in state:
            torch.cuda.set_rng_state_all(
                [value.detach().cpu().to(dtype=torch.uint8) for value in state['torch_cuda']]
            )
        loader_generator = getattr(self.train_loader, 'generator', None)
        if loader_generator is not None and 'train_loader_generator' in state:
            loader_generator.set_state(
                state['train_loader_generator'].detach().cpu().to(dtype=torch.uint8)
            )

    def _best_checkpoint_payload(self, epoch, val_loss, val_f1):
        return {
            'schema_version': self.checkpoint_schema_version,
            'checkpoint_role': 'best_validation_weighted_f1',
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'val_f1': val_f1,
            'selection_metric': 'validation_weighted_f1',
            'history': self.history,
            'metadata': self.checkpoint_metadata,
        }

    def save_resume_checkpoint(self, completed_epoch):
        payload = {
            'schema_version': self.checkpoint_schema_version,
            'checkpoint_role': 'epoch_boundary_resume',
            'completed_epoch': completed_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'best_model_path': str(self.best_model_path) if self.best_model_path else None,
            'patience_counter': self.patience_counter,
            'rng_state': self._rng_state(),
            'metadata': self.checkpoint_metadata,
        }
        self._atomic_torch_save(payload, self.resume_checkpoint_path)
        print(f"Epoch sınırı devam checkpoint'i kaydedildi: {self.resume_checkpoint_path}")

    def load_resume_checkpoint(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Devam checkpoint'i bulunamadı: {path}")
        # Load the archive on CPU first.  ``map_location=self.device`` also
        # moves CPU RNG and DataLoader-generator states to CUDA, where PyTorch's
        # RNG restoration APIs reject them.  Model and optimizer state loading
        # below copies/casts their tensors to the parameters' target device.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('checkpoint_role') != 'epoch_boundary_resume':
            raise ValueError(f"Geçersiz devam checkpoint rolü: {path}")
        expected_metadata = self.checkpoint_metadata
        actual_metadata = checkpoint.get('metadata', {})
        if expected_metadata and actual_metadata != expected_metadata:
            raise ValueError("Devam checkpoint metadata'sı mevcut çalıştırma ile eşleşmiyor")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint['best_val_f1']
        # The fixed filename is part of the v2 checkpoint contract. Rebuild the
        # path locally instead of trusting a container-specific absolute path.
        self.best_model_path = str(Path(self.output_dir) / "best_checkpoint.pt")
        if not Path(self.best_model_path).is_file():
            raise FileNotFoundError(
                f"Devam checkpoint'inin en iyi model dosyası bulunamadı: {self.best_model_path}"
            )
        self.patience_counter = checkpoint['patience_counter']
        self.start_epoch = int(checkpoint['completed_epoch']) + 1
        self._restore_rng_state(checkpoint.get('rng_state'))
        print(f"Devam checkpoint'i yüklendi: {path}; sonraki epoch={self.start_epoch + 1}")
    
    def train_epoch(self):
        """Bir epoch için eğit"""
        self.model.train()
        
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Eğitim")
        
        for batch in progress_bar:
            # Batch'i cihaza taşı
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Gradyanları sıfırla
            self.optimizer.zero_grad()
            
            # İleri geçiş
            logits = self.model(input_ids, attention_mask)
            
            # Kaybı hesapla
            loss = self.criterion(logits, labels)
            
            # Geri yayılım
            loss.backward()
            
            # Gradyan patlamasını önlemek için kırp
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Ağırlıkları güncelle
            self.optimizer.step()
            self.scheduler.step()
            
            # İlerleme çubuğunu güncelle
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Modeli doğrula"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Doğrulama")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # İleri geçiş
                logits = self.model(input_ids, attention_mask)
                
                # Kaybı hesapla
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Tahminleri al
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        
        # F1 skorunu hesapla
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, f1
    
    def train(self, early_stopping_patience=2):
        """
        Modeli eğit
        
        Args:
            early_stopping_patience: Erken durdurma öncesi beklenecek epoch sayısı
        """
        print(f"\nEğitim başlıyor: {self.device}")
        print(f"Epoch sayısı: {self.epochs}")
        print(f"Kayıp modu: {self.loss_mode}")
        print(f"Eğitim batch sayısı: {len(self.train_loader)}")
        print(f"Doğrulama batch sayısı: {len(self.val_loader)}")
        
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*50}")
            
            # Eğit
            train_loss = self.train_epoch()
            print(f"Ortalama eğitim kaybı: {train_loss:.4f}")
            
            # Doğrula
            val_loss, val_f1 = self.validate()
            print(f"Doğrulama kaybı: {val_loss:.4f}")
            print(f"Doğrulama F1 Skoru: {val_f1:.4f}")
            
            # Geçmişi kaydet
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            # En iyi modeli doğrulama F1 skoruna göre kaydet
            if val_f1 > self.best_val_f1:
                self.best_val_loss = val_loss
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                
                # Modeli kaydet
                model_filename = "best_checkpoint.pt"
                model_path = os.path.join(self.output_dir, model_filename)
                self._atomic_torch_save(
                    self._best_checkpoint_payload(epoch, val_loss, val_f1),
                    model_path,
                )
                
                self.best_model_path = model_path
                print(f"✓ En iyi model kaydedildi: {model_path}")
            else:
                self.patience_counter += 1
                print(f"İyileşme yok. Sabır: {self.patience_counter}/{early_stopping_patience}")

            self._write_history()
            self.save_resume_checkpoint(epoch)
            
            # Erken durdurma
            if self.patience_counter >= early_stopping_patience:
                print(f"\nErken durdurma tetiklendi: {epoch + 1} epoch sonra")
                break
        
        print(f"\n{'='*50}")
        print("Eğitim tamamlandı!")
        print(f"En iyi doğrulama kaybı: {self.best_val_loss:.4f}")
        print(f"En iyi doğrulama F1 Skoru: {self.best_val_f1:.4f}")
        print(f"En iyi model kaydedildi: {self.best_model_path}")
        print(f"{'='*50}\n")
        
        return self.history
    
    def load_best_model(self):
        """En iyi kaydedilmiş modeli yükle"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            checkpoint = torch.load(
                self.best_model_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"En iyi model yüklendi: {self.best_model_path}")
            return checkpoint
        else:
            raise FileNotFoundError("Kaydedilmiş en iyi model bulunamadı")

    def clear_resume_checkpoint(self):
        if self.resume_checkpoint_path.is_file():
            self.resume_checkpoint_path.unlink()


if __name__ == "__main__":
    print("Eğitim modülü hazır!")
