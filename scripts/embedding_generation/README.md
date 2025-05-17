# Medical Code Embedding Generation

This directory contains scripts for generating embeddings for medical codes (SNOMED, LOINC, RxNorm, CVX, etc.) found in the EHR data.

## Generate Code Embeddings

The `generate_code_embeddings.py` script uses BioBERT to create meaningful embeddings for medical codes that can be used in the GNN model. It processes all patient data files to extract unique codes and generates embeddings based on code descriptions.

### Usage

```bash
# Basic usage with default settings
python scripts/embedding_generation/generate_code_embeddings.py

# With custom settings
python scripts/embedding_generation/generate_code_embeddings.py \
  --data_dir data/processed_ehr_data \
  --output_dir embeddings \
  --model_name dmis-lab/biobert-base-cased-v1.1 \
  --batch_size 32
```

### Dependencies

Before running the script, make sure to install the required packages:

```bash
pip install torch transformers tqdm numpy
```

### Output

The script creates a structured directory for embeddings:

```
embeddings/
  ├── snomed/
  │   └── embeddings.pkl, embeddings.npz
  ├── loinc/
  │   └── embeddings.pkl, embeddings.npz
  ├── rxnorm/
  │   └── embeddings.pkl, embeddings.npz
  └── ... (other code systems)
```

Each embedding is generated based on the description of the medical code, which provides much richer semantic information than the code alone. 