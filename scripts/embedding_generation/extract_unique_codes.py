#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract and organize unique medical codes from EHR data.
This script processes all patient data files to extract unique codes by system,
tracks code usage by resource type, and saves the results to the data/codes/ directory.
"""

import os
import json
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Counter, DefaultDict
from collections import defaultdict, Counter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def collect_unique_codes(data_dir: str) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Counter]]]:
    """
    Scan through all data files to collect unique codes by system and track usage by resource type
    
    Args:
        data_dir: Directory containing processed EHR data files
        
    Returns:
        Tuple containing:
        - Dictionary mapping code systems to sets of unique codes
        - Dictionary mapping code systems to resource type usage counts
    """
    unique_codes = defaultdict(set)
    # Track resource types where each code system appears
    code_usage = defaultdict(lambda: defaultdict(Counter))
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
                    resource_type = event.get("resourceType", "Unknown")
                    
                    # Process primary codes (e.g., SNOMED, LOINC)
                    for code_entry in event.get("codes", []):
                        system = code_entry.get("system")
                        code = code_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                            code_usage[system][resource_type][code] += 1
                    
                    # Process status codes
                    for status_entry in event.get("status_codes", []):
                        system = status_entry.get("system")
                        code = status_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                            code_usage[system][resource_type][code] += 1
                    
                    # Process intent codes
                    for intent_entry in event.get("intent_codes", []):
                        system = intent_entry.get("system")
                        code = intent_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                            code_usage[system][resource_type][code] += 1
                    
                    # Process interpretation codes
                    for interp_code in event.get("interpretation_codes", []):
                        if isinstance(interp_code, str):
                            # If it's a direct string without system, treat it as an interpretation code
                            system = "interpretation_code"  # Generic system name instead of hardcoded URL
                            unique_codes[system].add(interp_code)
                            code_usage[system][resource_type][interp_code] += 1
                        elif isinstance(interp_code, dict) and interp_code.get("system") and interp_code.get("code"):
                            # If it's a system/code pair
                            system = interp_code.get("system")
                            code = interp_code.get("code")
                            unique_codes[system].add(code)
                            code_usage[system][resource_type][code] += 1
                    
                    # Process value_codeable_concept_code if present
                    if event.get("value_codeable_concept_code"):
                        code = event.get("value_codeable_concept_code")
                        # If we don't know the system, use the field name
                        system = "value_codeable_concept"
                        unique_codes[system].add(code)
                        code_usage[system][resource_type][code] += 1
                    
                    # Process unit_code if present
                    if event.get("unit_code"):
                        code = event.get("unit_code")
                        # If we don't know the system, use the field name
                        system = "unit_code"
                        unique_codes[system].add(code)
                        code_usage[system][resource_type][code] += 1
                
                # Process patient demographics (race and ethnicity codes)
                demographics = patient_data.get("demographics", {})
                for race_code in demographics.get("race_codes", []):
                    system = "race_code"
                    unique_codes[system].add(race_code)
                    code_usage[system]["Patient"][race_code] += 1
                    
                for ethnicity_code in demographics.get("ethnicity_codes", []):
                    system = "ethnicity_code"
                    unique_codes[system].add(ethnicity_code)
                    code_usage[system]["Patient"][ethnicity_code] += 1
                
                # Process gender
                gender = demographics.get("gender")
                if gender:
                    system = "gender"
                    unique_codes[system].add(gender)
                    code_usage[system]["Patient"][gender] += 1
                    
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Processed {file_count} files")
    for system, codes in unique_codes.items():
        logger.info(f"Found {len(codes)} unique codes for system {system}")
    
    return unique_codes, code_usage

def get_safe_filename(system: str) -> str:
    """
    Convert a system identifier to a safe filename
    
    Args:
        system: The original system identifier
        
    Returns:
        A filename-safe version of the system identifier
    """
    # Replace characters that are problematic in filenames
    return system.replace("/", "_").replace(":", "_").replace(".", "_").replace(" ", "_").lower()

def save_unique_codes(unique_codes: Dict[str, Set[str]], code_usage: Dict[str, Dict[str, Counter]], output_dir: str) -> None:
    """
    Save unique codes to CSV files, one file per code system.
    Also save code usage statistics.
    
    Args:
        unique_codes: Dictionary mapping code systems to sets of unique codes
        code_usage: Dictionary tracking code usage by resource type
        output_dir: Directory to save the code files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each code system to a separate file
    for system, codes in unique_codes.items():
        # Create a safe filename from the original system
        safe_system_name = get_safe_filename(system)
        output_file = os.path.join(output_dir, f"{safe_system_name}.csv")
        
        # Sort codes for consistent output
        sorted_codes = sorted(codes)
        
        # Save codes to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["system", "code"])
            for code in sorted_codes:
                writer.writerow([system, code])
        
        logger.info(f"Saved {len(sorted_codes)} codes to {output_file}")
        
        # Save detailed usage for this code system
        if system in code_usage:
            usage_file = os.path.join(output_dir, f"{safe_system_name}_usage.csv")
            with open(usage_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["system", "code", "resource_type", "count"])
                
                # For each resource type where this code system appears
                for resource_type, code_counts in code_usage[system].items():
                    # For each code and its count in this resource type
                    for code, count in code_counts.items():
                        writer.writerow([system, code, resource_type, count])
            
            logger.info(f"Saved usage statistics to {usage_file}")
    
    # Create a summary file with counts
    summary_file = os.path.join(output_dir, "code_summary.csv")
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["system", "code_count", "resource_types"])
        for system, codes in unique_codes.items():
            # Get the resource types where this code system appears
            resource_types = list(code_usage[system].keys()) if system in code_usage else []
            writer.writerow([system, len(codes), "|".join(resource_types)])
    
    logger.info(f"Saved summary to {summary_file}")
    
    # Create a detailed usage summary
    usage_summary_file = os.path.join(output_dir, "usage_summary.csv")
    with open(usage_summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["system", "resource_type", "unique_codes", "total_occurrences"])
        
        for system, resource_types in code_usage.items():
            for resource_type, code_counts in resource_types.items():
                unique_count = len(code_counts)
                total_occurrences = sum(code_counts.values())
                writer.writerow([system, resource_type, unique_count, total_occurrences])
    
    logger.info(f"Saved usage summary to {usage_summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract unique medical codes from EHR data")
    parser.add_argument(
        "--data_dir", 
        default="data/processed_ehr_data",
        help="Directory containing processed EHR data files"
    )
    parser.add_argument(
        "--output_dir", 
        default="data/codes",
        help="Directory to save extracted code files"
    )
    
    args = parser.parse_args()
    
    # Collect unique codes from all data files
    logger.info(f"Collecting unique codes from {args.data_dir}...")
    unique_codes, code_usage = collect_unique_codes(args.data_dir)
    
    # Save codes to files
    logger.info(f"Saving codes to {args.output_dir}...")
    save_unique_codes(unique_codes, code_usage, args.output_dir)
    
    logger.info("Code extraction complete")

if __name__ == "__main__":
    main() 