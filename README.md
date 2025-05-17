# EHR Snapshot Prediction with Temporal GNN and Diffusion Models

This project aims to predict the state of a patient's Electronic Health Record (EHR) one year into the future. It leverages a temporal Graph Neural Network (GNN) architecture combined with a diffusion model to learn from patient history and forecast future health snapshots.

## Project Overview

The core idea is to represent each patient's EHR data as a heterogeneous graph. This graph captures:
1.  **Patient Demographics**: Static information about the patient.
2.  **Annual Health Snapshots**: A chronological sequence of summarized health states for a patient, year by year.
3.  **Clinical Events**: Detailed medical occurrences (conditions, observations, medications, procedures, immunizations, diagnostic reports) within each annual snapshot.

The model processes this graph to generate an embedding representing the patient's health trajectory. A diffusion model then uses this embedding to predict the embedding of the subsequent year's health snapshot. Finally, a decoder translates this predicted embedding into potential new clinical events.

## Data

-   **Source**: Synthetic patient population data, where FHIR records have been processed into JSON files. Each JSON file (located in `data/processed_ehr_data/`) represents a single patient and contains their demographic information along with a chronologically ordered array of clinical events.
-   **Graph Structure Details**: The specific node types, their features, and the relationships (edges) between them are detailed in `docs/graph_structure.md`.
-   **Data Handling**: The `data/` directory (containing raw and processed patient data) is excluded from Git version control via `.gitignore`.

## Key Components

1.  **Embedding Generation (`scripts/embedding_generation/`)**:
    *   Medical codes (SNOMED CT, LOINC, RxNorm, CVX, etc.) are crucial features for the GNN.
    *   The script `scripts/embedding_generation/generate_code_embeddings.py` is responsible for:
        *   Extracting unique medical codes from all patient data files.
        *   Generating dense vector embeddings for these codes using pre-trained biomedical NLP models like BioBERT. These embeddings capture the semantic meaning of the codes.
        *   Storing the generated embeddings in the `embeddings/` directory, organized by code system (e.g., `embeddings/snomed/`, `embeddings/loinc/`).
    *   The `embeddings/` directory is also excluded from Git.

2.  **Graph Construction**:
    *   Scripts (to be developed) will transform the patient JSON data and the pre-computed code embeddings into a graph format suitable for PyTorch Geometric.

3.  **Modeling Pipeline**:
    *   **Intra-Snapshot Attention**: An attention mechanism will be applied to the clinical events within each `AnnualSnapshot` to create a summary embedding for that year.
    *   **Temporal Attention**: A second attention mechanism will operate over the sequence of `AnnualSnapshot` embeddings to learn a comprehensive patient history embedding.
    *   **Patient Embedding**: The patient history embedding will be combined with an embedding derived from the static patient demographic data.
    *   **Diffusion Model**: This combined patient embedding will serve as the input to a diffusion model tasked with predicting the embedding of the next year's health snapshot.
    *   **Decoder**: A decoder module will interpret the predicted embedding to output predictions, initially focusing on the classes of new clinical events expected in the following year.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ehr_prediction
    ```
2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```
3.  **Install dependencies:**
    The project dependencies are listed in `requirements.txt`. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
    Key packages include PyTorch, PyTorch Geometric, and Transformers (for BioBERT).

## Usage

1.  **Generate Code Embeddings**:
    Before training any models, you need to generate the embeddings for the medical codes:
    ```bash
    python scripts/embedding_generation/generate_code_embeddings.py
    ```
    This will populate the `embeddings/` directory. Refer to `scripts/embedding_generation/README.md` for more details and options.

2.  **(Future Steps)**
    *   Run graph construction scripts.
    *   Train the GNN and diffusion models.
    *   Evaluate prediction performance.

## Directory Structure

```
.
├── data/                     # (Git ignored) Patient EHR data (raw and processed)
│   └── processed_ehr_data/   # JSON files per patient
├── docs/                     # Project documentation
│   └── graph_structure.md    # Detailed schema for graph nodes and edges
├── embeddings/               # (Git ignored) Generated code embeddings
├── scripts/                  # Utility and processing scripts
│   └── embedding_generation/ # Scripts for generating code embeddings
│       ├── generate_code_embeddings.py
│       └── README.md
├── .gitignore                # Specifies intentionally untracked files by Git
├── README.md                 # This file
└── requirements.txt          # Python package dependencies
```

## Contributing

(Details to be added as the project evolves)

## License

(Specify license if applicable) 