from pathlib import Path

def get_raw_data(project_dir) -> Path:
    return project_dir / 'data/raw'

def get_interim_data(project_dir) -> Path:
    return project_dir / 'data/interim'
