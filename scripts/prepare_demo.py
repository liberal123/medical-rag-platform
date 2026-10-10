"""Prepare the authored workflow illustration; reuse identical output safely."""
import hashlib
from pathlib import Path
from medical_rag.config import Settings


def main():
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_env()
    source = root / 'examples' / 'workflow.svg'
    target = settings.image_dir / 'workflow.svg'
    target.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(content).digest():
            raise FileExistsError('Existing illustration differs; choose another MEDRAG_DATA_DIR.')
    else:
        target.write_bytes(content)
    print('Demo illustration ready. No medical documents or model weights were downloaded.')


if __name__ == '__main__':
    main()
