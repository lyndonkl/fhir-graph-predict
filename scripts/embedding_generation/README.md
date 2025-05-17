# Medical Code Extraction

This directory contains scripts for extracting and organizing medical codes found in the EHR data.

## Extract Unique Codes

The `extract_unique_codes.py` script extracts and organizes the unique codes from the patient data files, and tracks their usage by resource type and source field. This provides a comprehensive view of how different codes are used across the dataset.

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
  ├── snomed_usage.csv           # Detailed usage of SNOMED codes by resource type and source field
  ├── loinc.csv                  # List of unique LOINC codes
  ├── loinc_usage.csv            # Detailed usage of LOINC codes by resource type and source field
  ├── ...                        # Other code systems
  ├── code_summary.csv           # Summary of all code systems with counts and context
  └── usage_summary.csv          # Detailed statistics by system, resource type, and source field
```

#### Main code files (e.g., `snomed.csv`):
- `system`: The original system identifier
- `code`: The unique code value

#### Usage files (e.g., `snomed_usage.csv`):
- `system`: The original system identifier
- `code`: The code value
- `resource_type`: The FHIR resource type where this code appears (Condition, Observation, etc.)
- `source_field`: The field in the data where this code was found (codes, status_codes, intent_codes, etc.)
- `count`: The number of occurrences of this code in this context

#### Summary files:
- `code_summary.csv`: Lists each code system, the count of unique codes, the resource types where it appears, and the source fields where it's used
- `usage_summary.csv`: Provides detailed statistics including system, resource type, source field, unique codes, and total occurrences

### Implementation Notes

1. **No Hard-Coded Identifiers**: The script avoids hard-coding system identifiers, instead using information directly from the data.

2. **Consistent Field Names**: When a system identifier isn't available (e.g., for interpretation codes), the script uses the field name in a consistent way.

3. **Complete Usage Context Tracking**: The script tracks every occurrence of each code with full context:
   - Which resource type it appears in (e.g., Condition, Observation)
   - Which field it comes from (e.g., codes, status_codes, interpretation_codes)
   - How frequently it occurs in each context

### Planned Next Steps

After collecting and organizing the codes, we plan to:

1. Analyze the usage statistics to understand code distribution across resource types and source fields
2. For medical code systems (SNOMED, LOINC, RxNorm, etc.), fetch official descriptions where possible
3. Generate embeddings using biomedical NLP models like BioBERT, potentially tailored by resource type and source field context 