"""
Türkçe Duygu Analizi için Veri Yükleme ve Ön İşleme
"""
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.protocol_constants import (
    BERTURK_TOKENIZER_ID,
    BERTURK_TOKENIZER_REVISION,
    WINVOKER_ASSIGNMENTS_PATH,
    WINVOKER_ASSIGNMENTS_SHA256,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
)
from src.preprocessing import apply_preprocessing_profile, sha256_file


HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
WHITESPACE_RE = re.compile(r"\s+")
# ! ve ? gibi duygu ile ilgili noktalama işaretlerini korurken, sadece noktalama işaretlerinden oluşan gürültü parçalarını temizle.
PUNCT_NOISE_RE = re.compile(r"(?u)(?<!\w)[^\w\s!?]+(?!\w)")
DEFAULT_LABEL_MAPPING = {'Positive': 0, 'Negative': 1, 'Notr': 2}


class TurkishSentimentDataset(Dataset):
    """Türkçe Duygu Analizi için Özel Veri Seti"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_winvoker_dataset(
    dataset_name=WINVOKER_DATASET_ID,
    revision=WINVOKER_DATASET_REVISION,
):
    """Winvoker Türkçe Duygu Analizi veri setini Hugging Face'den yükle"""
    print(f"Winvoker veri setini yüklüyor: {dataset_name}@{revision}")
    dataset = load_dataset(dataset_name, revision=revision)
    return dataset


def clean_text(text):
    """
    Metni makale iddialarıyla uyumlu şekilde temizle.

    Yapılan işlemler:
    - Baştaki/sondaki boşlukları temizle
    - HTML tag'lerini kaldır
    - URL'leri kaldır
    - @mention ifadelerini kaldır
    - Yalnızca anlamsız noktalama gürültüsünü kaldır
    - Duygu sinyali taşıyabilen ! ve ? işaretlerini koru
    """
    text = str(text).strip()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = PUNCT_NOISE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess_dataset(dataset, test_size=0.2, val_size=0.1, random_state=42):
    """
    Veri setini ön işleme ve bölme
    
    Args:
        dataset: Hugging Face dataset nesnesi
        test_size: Test seti için veri kesimi
        val_size: Eğitim veri seti için doğrulama seti
        random_state: Tekrarlanabilirlik için rastgele seed
    """
    # pandas için daha kolay işleme
    df = pd.DataFrame(dataset['train'])
    
    # Temizlik
    df['text'] = df['text'].apply(clean_text)
    
    # Etiketleri sayısal değerlerle eşleme
    label_mapping = DEFAULT_LABEL_MAPPING
    df['label'] = df['label'].map(label_mapping)
    
    # Sınıf dağılımı
    print("\nSınıf dağılımı:")
    print(df['label'].value_counts(normalize=True))
    print(f"\nToplam örnek: {len(df)}")
    
    # Eğitim + Doğrulama ve Test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # Eğitim + Doğrulama bölmesini eğitim ve doğrulama olarak ayır
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['label']
    )
    
    print(f"\nEğitim örnek sayısı: {len(train_df)}")
    print(f"Doğrulama örnek sayısı: {len(val_df)}")
    print(f"Test örnek sayısı: {len(test_df)}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def preprocess_dataset_with_profile(
    dataset,
    profile,
    included_splits=("train", "val", "test"),
    assignments_path=WINVOKER_ASSIGNMENTS_PATH,
):
    """Build locked split views and clean only the explicitly included splits.

    The assignment artifact fixes membership and order independently of the
    preprocessing profile. A validation-only screening call passes only
    ``("train", "val")``, so no cleaned test frame is constructed.
    """

    normalized = tuple(str(value) for value in included_splits)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("included_splits must be a non-empty set of unique split names")
    invalid = sorted(set(normalized) - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"Unknown included split names: {invalid}")
    assignments_path = Path(assignments_path)
    if sha256_file(assignments_path) != WINVOKER_ASSIGNMENTS_SHA256:
        raise ValueError("Locked split-assignment hash mismatch")

    upstream = dataset["train"]
    assignments = pd.read_csv(assignments_path, keep_default_na=False)
    upstream_required = {"text", "label", "dataset"}
    assignment_required = {
        "split",
        "order_in_split",
        "source_row_index",
        "label",
        "label_id",
        "dataset",
    }
    if missing := upstream_required - set(upstream.column_names):
        raise ValueError(f"Winvoker upstream data is missing columns: {sorted(missing)}")
    if missing := assignment_required - set(assignments):
        raise ValueError(f"Split assignments are missing columns: {sorted(missing)}")

    name_map = {"train": "train", "val": "validation", "test": "test"}
    result = {}
    for split_name in normalized:
        assignment_name = name_map[split_name]
        rows = assignments.loc[assignments["split"].eq(assignment_name)].copy()
        positions = pd.to_numeric(rows["order_in_split"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(positions, np.arange(len(rows), dtype=np.int64)):
            raise ValueError(f"Non-contiguous locked ordering for {assignment_name}")
        source_rows = pd.to_numeric(rows["source_row_index"], errors="raise").to_numpy(
            dtype=np.int64
        )
        # Materialize only the permitted row IDs. In screening mode no held-out
        # test row is selected or converted to a pandas frame.
        frame = pd.DataFrame(upstream.select(source_rows.tolist())).reset_index(drop=True)
        if not np.array_equal(frame["label"].astype(str), rows["label"].astype(str)):
            raise ValueError(f"Label mismatch in locked {assignment_name} split")
        if not np.array_equal(frame["dataset"].astype(str), rows["dataset"].astype(str)):
            raise ValueError(f"Source mismatch in locked {assignment_name} split")
        frame["source_row_index"] = source_rows
        frame["order_in_split"] = positions
        frame["text"] = [apply_preprocessing_profile(value, profile) for value in frame["text"]]
        frame["label"] = frame["label"].map(DEFAULT_LABEL_MAPPING)
        if frame["label"].isna().any():
            raise ValueError(f"Unknown label in locked {assignment_name} split")
        frame["label"] = frame["label"].astype(np.int64)
        expected_ids = pd.to_numeric(rows["label_id"], errors="raise").to_numpy(dtype=np.int64)
        if not np.array_equal(frame["label"].to_numpy(dtype=np.int64), expected_ids):
            raise ValueError(f"Label-ID mismatch in locked {assignment_name} split")
        result[split_name] = frame
    return result


def calculate_class_weights(labels):
    """
    Sınıf ağırlıklarını hesapla
    
    Args:
        labels: Etiketler dizisi
    
    Returns:
        Sınıf ağırlıkları
    """
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)


def create_data_loaders(data_splits, tokenizer, batch_size=32, max_length=128, seed=42):
    """
    PyTorch DataLoaders için train, val ve test setleri oluştur
    
    Args:
        data_splits: Dictionary with train, val, test dataframes
        tokenizer: BERT tokenizer
        batch_size: DataLoader için batch size
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with train, val, test DataLoaders and class weights
    """
    datasets = {}
    loaders = {}
    
    for split_name, df in data_splits.items():
        datasets[split_name] = TurkishSentimentDataset(
            texts=df['text'].values,
            labels=df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        shuffle = (split_name == 'train')
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
        loaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Windows uyumluluğu için
            generator=generator,
        )
    
    # Eğitim setinden sınıf ağırlıklarını hesapla
    class_weights = calculate_class_weights(data_splits['train']['label'].values)
    
    return {
        'loaders': loaders,
        'class_weights': class_weights,
        'label_mapping': {0: 'Positive', 1: 'Negative', 2: 'Notr'}
    }


if __name__ == "__main__":
    # Veri yükleme testi
    print("Veri yükleme testi...")
    
    # Tokenizer yükleme
    tokenizer = AutoTokenizer.from_pretrained(
        BERTURK_TOKENIZER_ID,
        revision=BERTURK_TOKENIZER_REVISION,
    )
    
    # Veri setini yükle ve ön işleme
    dataset = load_winvoker_dataset()
    data_splits = preprocess_dataset(dataset)
    
    # DataLoaders oluştur
    data_config = create_data_loaders(data_splits, tokenizer, batch_size=32)
    
    print(f"\nSınıf ağırlıkları: {data_config['class_weights']}")
    
    # Test etme için train loader:
    print("\nTest etme için train loader:")
    for batch in data_config['loaders']['train']:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break
    
    print("\nData loader hazır!")
