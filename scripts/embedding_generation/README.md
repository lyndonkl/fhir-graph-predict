# Medical Code Extraction

This directory contains scripts for extracting and organizing medical codes found in the EHR data.

## Extract Unique Codes

The `extract_unique_codes.py` script extracts and organizes the unique codes from the patient data files, and tracks their usage by resource type. This provides a comprehensive view of how different codes are used across the dataset.

### Usage

```bash
# Basic usage with default settings
python scripts/embedding_generation/extract_unique_codes.py

# With custom settings
python scripts/embedding_generation/extract_unique_codes.py \
  --data_dir data/processed_ehr_data \
  --output_dir data/codes
```

### Dependencies

Before running the script, make sure to install the required package:

```bash
pip install tqdm
```

### Output

The script generates multiple types of CSV files in the output directory:

```
data/codes/
  ├── snomed.csv                 # List of unique SNOMED codes
  ├── snomed_usage.csv           # Detailed usage of SNOMED codes by resource type
  ├── loinc.csv                  # List of unique LOINC codes
  ├── loinc_usage.csv            # Detailed usage of LOINC codes by resource type
  ├── ...                        # Other code systems
  ├── code_summary.csv           # Summary of all code systems with counts
  └── usage_summary.csv          # Detailed usage statistics by code system and resource type
```

#### Main code files (e.g., `snomed.csv`):
- `system`: The original system identifier
- `code`: The unique code value

#### Usage files (e.g., `snomed_usage.csv`):
- `system`: The original system identifier
- `code`: The code value
- `resource_type`: The FHIR resource type where this code appears (Condition, Observation, etc.)
- `count`: The number of occurrences of this code in this resource type

#### Summary files:
- `code_summary.csv`: Lists each code system, the count of unique codes, and the resource types where it appears
- `usage_summary.csv`: Provides detailed statistics on code usage by system and resource type

### Implementation Notes

1. **No Hard-Coded Identifiers**: The script avoids hard-coding system identifiers, instead using information directly from the data.

2. **Consistent Field Names**: When a system identifier isn't available (e.g., for interpretation codes), the script uses the field name in a consistent way.

3. **Usage Tracking**: The script tracks every occurrence of each code, organized by resource type, providing valuable context for embedding generation.

### Planned Next Steps

After collecting and organizing the codes, we plan to:

1. Analyze the usage statistics to understand code distribution across resource types
2. For medical code systems (SNOMED, LOINC, RxNorm, etc.), fetch official descriptions where possible
3. Generate embeddings using biomedical NLP models like BioBERT, potentially tailoring them by resource type 