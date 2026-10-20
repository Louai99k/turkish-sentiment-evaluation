"""Locked identifiers shared by protocol-v2 experiment infrastructure."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

EXPERIMENT_SCHEMA_VERSION = "experiment-artifacts-v2.0"
METRICS_SCHEMA_VERSION = "classification-metrics-v2.0"
PREDICTIONS_SCHEMA_VERSION = "predictions-v2.0"

LABEL_IDS = (0, 1, 2)
ID_TO_LABEL = {0: "Positive", 1: "Negative", 2: "Notr"}
LABEL_TO_ID = {value: key for key, value in ID_TO_LABEL.items()}

WINVOKER_DATASET_ID = "winvoker/turkish-sentiment-analysis-dataset"
WINVOKER_DATASET_REVISION = "4bd9103ad5f24d44db806d40c94c8b9ec116ad05"
WINVOKER_UPSTREAM_SPLIT = "train"
WINVOKER_SPLIT_SEED = 42
WINVOKER_TEST_ROWS = 88_136
WINVOKER_STRICT_TEST_ROWS = 87_632
WINVOKER_TEST_SPLIT_ID = "winvoker-historic-split42-test"

WINVOKER_ASSIGNMENTS_PATH = (
    ROOT_DIR / "outputs" / "audits" / "historic_split_lineage_v1" / "split_assignments.csv"
)
WINVOKER_ASSIGNMENTS_SHA256 = (
    "085af50ef3c28c2b50696d5cb2dfb6f0d95289eeed7ffb65f2b4315778526839"
)
WINVOKER_TEST_MASK_PATH = (
    ROOT_DIR / "outputs" / "audits" / "historic_split_lineage_v1" / "test_inclusion_mask.csv"
)
WINVOKER_TEST_MASK_SHA256 = (
    "635f28535111efec1baeae4b9ba2e3f28b7dde8faa6240cf4b9afdad86de5999"
)

BERTURK_MODEL_ID = "dbmdz/bert-base-turkish-cased"
BERTURK_MODEL_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"
BERTURK_TOKENIZER_ID = BERTURK_MODEL_ID
BERTURK_TOKENIZER_REVISION = BERTURK_MODEL_REVISION

CANONICAL_SOURCES = (
    "HUMIR",
    "magaza_yorumlari",
    "random",
    "tweet-pn",
    "urun_yorumlari",
    "wiki",
)
