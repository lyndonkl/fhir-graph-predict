#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate embeddings for medical codes (SNOMED, LOINC, RxNorm, CVX) using BioBERT.
This script processes all patient data files to extract unique codes, generates embeddings,
and saves them in the embeddings directory.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dictionary mapping medical code systems to their canonical names
CODE_SYSTEMS = {
    "SNOMED": "snomed",
    "LOINC": "loinc",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "rxnorm",
    "http://hl7.org/fhir/sid/cvx": "cvx",
    "http://www.ama-assn.org/go/cpt": "cpt",
    "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation": "interpretation",
    "http://unitsofmeasure.org": "unit",
    "condition-clinical": "condition_clinical_status",
    "condition-verification": "condition_verification_status",
    "diagnosticreport-status": "diagnosticreport_status",
    "medicationrequest-status": "medication_status",
    "medicationrequest-intent": "medication_intent",
    "procedure-status": "procedure_status",
    "immunization-status": "immunization_status"
}

# Map of code systems to their official name (used for fetching descriptions)
CODE_SYSTEM_OFFICIAL_NAMES = {
    "SNOMED": "SNOMED CT",
    "LOINC": "LOINC",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
    "http://hl7.org/fhir/sid/cvx": "CVX"
}

# Medical code systems that typically have detailed descriptions for embedding
MEDICAL_CODE_SYSTEMS = ["SNOMED", "LOINC", "http://www.nlm.nih.gov/research/umls/rxnorm", "http://hl7.org/fhir/sid/cvx"]

# Fallback descriptions for codes where API calls might not work or are unnecessary
FALLBACK_DESCRIPTIONS = {
    # Status codes
    "condition-clinical": {
        "active": "Condition is currently active",
        "recurrence": "Condition has recurred",
        "relapse": "Condition has relapsed",
        "inactive": "Condition is inactive",
        "remission": "Condition is in remission",
        "resolved": "Condition has been resolved"
    },
    "condition-verification": {
        "unconfirmed": "Condition is unconfirmed",
        "provisional": "Condition is provisional",
        "differential": "Condition is differential",
        "confirmed": "Condition is confirmed",
        "refuted": "Condition is refuted",
        "entered-in-error": "Condition was entered in error"
    },
    "diagnosticreport-status": {
        "registered": "Diagnostic report is registered",
        "partial": "Diagnostic report is partial",
        "preliminary": "Diagnostic report is preliminary",
        "final": "Diagnostic report is final",
        "amended": "Diagnostic report is amended",
        "corrected": "Diagnostic report is corrected",
        "appended": "Diagnostic report is appended",
        "cancelled": "Diagnostic report is cancelled",
        "entered-in-error": "Diagnostic report was entered in error",
        "unknown": "Diagnostic report status is unknown"
    },
    "procedure-status": {
        "preparation": "Procedure is in preparation",
        "in-progress": "Procedure is in progress",
        "not-done": "Procedure was not done",
        "on-hold": "Procedure is on hold",
        "stopped": "Procedure was stopped",
        "completed": "Procedure was completed",
        "entered-in-error": "Procedure was entered in error",
        "unknown": "Procedure status is unknown"
    },
    "medicationrequest-status": {
        "active": "Medication request is active",
        "on-hold": "Medication request is on hold",
        "cancelled": "Medication request was cancelled",
        "completed": "Medication request was completed",
        "entered-in-error": "Medication request was entered in error",
        "stopped": "Medication request was stopped",
        "draft": "Medication request is draft",
        "unknown": "Medication request status is unknown"
    },
    "medicationrequest-intent": {
        "proposal": "Medication request is a proposal",
        "plan": "Medication request is a plan",
        "order": "Medication request is an order",
        "original-order": "Medication request is an original order",
        "reflex-order": "Medication request is a reflex order",
        "filler-order": "Medication request is a filler order",
        "instance-order": "Medication request is an instance order"
    },
    "immunization-status": {
        "completed": "Immunization was completed",
        "entered-in-error": "Immunization was entered in error",
        "not-done": "Immunization was not done"
    },
    # Observation interpretation
    "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation": {
        "L": "Below low normal",
        "H": "Above high normal",
        "LL": "Below lower critical limit",
        "HH": "Above upper critical limit",
        "N": "Normal",
        "<": "Off scale low",
        ">": "Off scale high",
        "A": "Abnormal",
        "AA": "Critically abnormal",
        "U": "Significant change up",
        "D": "Significant change down",
        "B": "Better",
        "W": "Worse"
    }
}

def mean_pooling(token_embeddings, attention_mask):
    """
    Perform mean pooling on token embeddings using attention mask
    """
    # Sum token embeddings and count tokens with attention
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Calculate mean by dividing sum by count
    return sum_embeddings / sum_mask

class CodeEmbeddingGenerator:
    """
    Generate embeddings for medical codes using BioBERT
    """
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        """
        Initialize with the pre-trained BioBERT model
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Dictionary to cache code descriptions to avoid repeated API calls
        self.code_descriptions = {}
    
    def get_code_description(self, system: str, code: str) -> str:
        """
        Get description for a medical code. For simplicity, this implementation uses
        fallback descriptions where available or generates a simple description.
        
        In a production environment, this would likely call external APIs like:
        - UMLS API for SNOMED CT
        - LOINC API for LOINC codes
        - RxNorm API for RxNorm codes
        - CDC API or a lookup table for CVX codes
        
        Returns:
            Description of the code as a string
        """
        # Check if description is already in cache
        cache_key = f"{system}:{code}"
        if cache_key in self.code_descriptions:
            return self.code_descriptions[cache_key]
        
        # Check if we have fallback descriptions for this system
        if system in FALLBACK_DESCRIPTIONS and code in FALLBACK_DESCRIPTIONS[system]:
            description = FALLBACK_DESCRIPTIONS[system][code]
        else:
            # For this implementation, we'll just use system:code as a fallback description
            # In a real-world scenario, you would implement API calls to fetch actual descriptions
            system_name = CODE_SYSTEM_OFFICIAL_NAMES.get(system, system)
            description = f"{system_name} code {code}"
        
        # Cache and return description
        self.code_descriptions[cache_key] = description
        return description
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a given text using BioBERT
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Numpy array of the embedding
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs)
        
        # Mean pooling to get sentence embedding
        embeddings = mean_pooling(model_output.last_hidden_state, inputs["attention_mask"])
        
        # Return as numpy array
        return embeddings.cpu().numpy()[0]

def collect_unique_codes(data_dir: str) -> Dict[str, Set[str]]:
    """
    Scan through all data files to collect unique codes by system
    
    Args:
        data_dir: Directory containing processed EHR data files
        
    Returns:
        Dictionary mapping code systems to sets of unique codes
    """
    unique_codes = defaultdict(set)
    file_count = 0
    
    # List all JSON files in data directory
    json_files = list(Path(data_dir).glob("*.json"))
    
    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")
    
    # Process each file
    for file_path in tqdm(json_files, desc="Collecting unique codes"):
        with open(file_path, "r") as f:
            try:
                patient_data = json.load(f)
                file_count += 1
                
                # Process clinical events
                for event in patient_data.get("clinical_events", []):
                    # Process primary codes (e.g., SNOMED, LOINC)
                    for code_entry in event.get("codes", []):
                        system = code_entry.get("system")
                        code = code_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                    
                    # Process status codes
                    for status_entry in event.get("status_codes", []):
                        system = status_entry.get("system")
                        code = status_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                    
                    # Process intent codes
                    for intent_entry in event.get("intent_codes", []):
                        system = intent_entry.get("system")
                        code = intent_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                    
                    # Process interpretation codes (these are usually strings in the sample data)
                    for interp_code in event.get("interpretation_codes", []):
                        if isinstance(interp_code, str):
                            unique_codes["http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"].add(interp_code)
                
                # Process patient demographics (race and ethnicity codes)
                demographics = patient_data.get("demographics", {})
                for race_code in demographics.get("race_codes", []):
                    unique_codes["race"].add(race_code)
                for ethnicity_code in demographics.get("ethnicity_codes", []):
                    unique_codes["ethnicity"].add(ethnicity_code)
                
                # Process gender
                gender = demographics.get("gender")
                if gender:
                    unique_codes["gender"].add(gender)
                    
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Processed {file_count} files")
    for system, codes in unique_codes.items():
        logger.info(f"Found {len(codes)} unique codes for system {system}")
    
    return unique_codes

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for medical codes using BioBERT")
    parser.add_argument(
        "--data_dir", 
        default="data/processed_ehr_data",
        help="Directory containing processed EHR data files"
    )
    parser.add_argument(
        "--output_dir", 
        default="embeddings",
        help="Directory to save embeddings"
    )
    parser.add_argument(
        "--model_name", 
        default="dmis-lab/biobert-base-cased-v1.1",
        help="HuggingFace model name/path for the BioBERT model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for processing embeddings"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories for each code system
    for system_key in CODE_SYSTEMS.values():
        os.makedirs(os.path.join(args.output_dir, system_key), exist_ok=True)
    
    # Collect unique codes from all data files
    logger.info(f"Collecting unique codes from {args.data_dir}...")
    unique_codes = collect_unique_codes(args.data_dir)
    
    # Initialize embedding generator
    embedding_generator = CodeEmbeddingGenerator(model_name=args.model_name)
    
    # Generate and save embeddings for each code system
    total_embeddings = 0
    
    for system, codes in unique_codes.items():
        logger.info(f"Generating embeddings for {len(codes)} {system} codes...")
        
        # Determine output file
        output_subdir = CODE_SYSTEMS.get(system, system.lower().replace(" ", "_").replace("-", "_"))
        output_path = os.path.join(args.output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        
        # Dictionary to store embeddings for this system
        system_embeddings = {}
        
        # Generate embeddings in batches
        codes_list = list(codes)
        for i in tqdm(range(0, len(codes_list), args.batch_size), 
                      desc=f"Generating {system} embeddings"):
            batch_codes = codes_list[i:i + args.batch_size]
            
            for code in batch_codes:
                # Get description for the code
                if system in MEDICAL_CODE_SYSTEMS:
                    description = embedding_generator.get_code_description(system, code)
                else:
                    # For non-medical systems, use simpler descriptions
                    description = f"{system} code: {code}"
                
                # Generate embedding
                embedding = embedding_generator.generate_embedding(description)
                
                # Store embedding
                system_embeddings[code] = embedding
                total_embeddings += 1
        
        # Save embeddings for this system
        embeddings_file = os.path.join(output_path, "embeddings.pkl")
        with open(embeddings_file, "wb") as f:
            pickle.dump(system_embeddings, f)
        
        # Also save as numpy array for easier loading in some contexts
        embeddings_file = os.path.join(output_path, "embeddings.npz")
        np.savez(
            embeddings_file,
            codes=codes_list,
            embeddings=np.array([system_embeddings[code] for code in codes_list])
        )
        
        logger.info(f"Saved {len(system_embeddings)} embeddings to {output_path}")
    
    logger.info(f"Generated a total of {total_embeddings} embeddings")

if __name__ == "__main__":
    main() 