"""Centralized path configuration for LungLens V5 scripts.

Environment variables:
  LUNGLENS_PROJECT_ROOT (optional): project root; defaults to parent of scripts/
  LUNGLENS_DATA_ROOT (optional): LUNA16 data root; defaults to E:\\LungLens_Data
  LUNGLENS_EVAL_SPLIT (optional): fixed evaluation split CSV; defaults to V4.1

The legacy evaluation split is intentional: V4.1 and V5 originally used identical
11 validation and 7 test cases. Do not change the cohort when comparing results.
"""
import os
from pathlib import Path


def path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, '').strip()
    return Path(value).expanduser() if value else default


PROJECT_ROOT = path_from_env('LUNGLENS_PROJECT_ROOT', Path(__file__).resolve().parent.parent)
LUNA_ROOT = path_from_env('LUNGLENS_DATA_ROOT', Path(r'E:\LungLens_Data'))
RAW_ROOT = LUNA_ROOT / 'raw'
ANNOTATIONS_DIR = LUNA_ROOT / 'annotations'
ANNOTATIONS_FILE = ANNOTATIONS_DIR / 'annotations.csv'
EXCLUDED_FILE = ANNOTATIONS_DIR / 'annotations_excluded.csv'
DATA_ROOT = LUNA_ROOT / 'processed' / 'yolo_v5_dataset'
CANONICAL_SPLIT = LUNA_ROOT / 'processed' / 'yolo_v5' / 'case_split.csv'
LEGACY_EVAL_SPLIT = path_from_env(
    'LUNGLENS_EVAL_SPLIT',
    LUNA_ROOT / 'processed' / 'yolo_v41' / 'case_split.csv',
)

if __name__ == '__main__':
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('LUNA_ROOT:', LUNA_ROOT)
    print('CANONICAL_SPLIT:', CANONICAL_SPLIT)
    print('LEGACY_EVAL_SPLIT:', LEGACY_EVAL_SPLIT)
