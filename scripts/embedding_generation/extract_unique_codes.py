#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract and organize unique medical codes from EHR data.
This script processes all patient data files to extract unique codes by system,
tracks code usage by resource type and source field (when system is obtained dynamically),
and saves the results to the data/codes/ directory.
"""

import os
import json
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Counter, DefaultDict, NamedTuple, Optional
from collections import defaultdict, Counter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define a namedtuple to track source field for each code
class CodeSource(NamedTuple):
    resource_type: str
    source_field: Optional[str] = None  # Optional as we'll only track source for dynamic systems

def collect_unique_codes(data_dir: str) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[CodeSource, Counter]]]:
    """
    Scan through all data files to collect unique codes by system and track usage by resource type and source field
    
    Args:
        data_dir: Directory containing processed EHR data files
        
    Returns:
        Tuple containing:
        - Dictionary mapping code systems to sets of unique codes
        - Dictionary mapping code systems to usage counts by resource type and source field
    """
    unique_codes = defaultdict(set)
    # Track resource types and source fields where each code system appears
    # Structure: code_usage[system][CodeSource(resource_type, source_field)][code] = count
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
                    for code_entry in event.get("primary_codings", []):
                        system = code_entry.get("system")  # System is obtained dynamically
                        code = code_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                            # Track source field since system is obtained dynamically
                            source = CodeSource(resource_type, "primary_codings")
                            code_usage[system][source][code] += 1
                    
                    # Process status codes
                    for status_entry in event.get("status_codings", []):
                        system = status_entry.get("system")  # System is obtained dynamically
                        code = status_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                            # Track source field since system is obtained dynamically
                            source = CodeSource(resource_type, "status_codings")
                            code_usage[system][source][code] += 1
                    
                    # Process intent codes
                    for intent_entry in event.get("intent_codings", []):
                        system = intent_entry.get("system")  # System is obtained dynamically
                        code = intent_entry.get("code")
                        if system and code:
                            unique_codes[system].add(code)
                            # Track source field since system is obtained dynamically
                            source = CodeSource(resource_type, "intent_codings")
                            code_usage[system][source][code] += 1
                    
                    # Process interpretation codes
                    for interp_entry in event.get("interpretation_codings", []):
                        # Each interp_entry is now a dict with 'system' and 'code' keys
                        if isinstance(interp_entry, dict) and interp_entry.get("system") and interp_entry.get("code"):
                            system = interp_entry.get("system") 
                            code = interp_entry.get("code")
                            unique_codes[system].add(code)
                            # System is now dynamically obtained from interp_entry
                            source = CodeSource(resource_type, "interpretation_codings") 
                            code_usage[system][source][code] += 1
                    
                    # Process value_concept_codings (plural)
                    for value_entry in event.get("value_concept_codings", []): # Iterate over the list
                        # Each value_entry is now a dict with 'system' and 'code' keys
                        if isinstance(value_entry, dict) and value_entry.get("system") and value_entry.get("code"):
                            system = value_entry.get("system")
                            code = value_entry.get("code")
                            unique_codes[system].add(code)
                            # System is now dynamically obtained from value_entry
                            source = CodeSource(resource_type, "value_concept_codings")
                            code_usage[system][source][code] += 1
                    
                    # Process unit_code if present
                    if event.get("unit_code"):
                        code = event.get("unit_code")
                        # Hard-coded system name
                        system = "unit_code"
                        unique_codes[system].add(code)
                        # Don't track source field for hard-coded system name
                        source = CodeSource(resource_type)
                        code_usage[system][source][code] += 1
                
                # Process patient demographics (race and ethnicity codes)
                demographics = patient_data.get("demographics", {})
                for race_entry in demographics.get("race_codings", []):
                    if isinstance(race_entry, dict) and race_entry.get("code"):
                        race_code = race_entry.get("code")
                        # Hard-coded system name
                        system = "race_code"
                        unique_codes[system].add(race_code)
                        # Don't track source field for hard-coded system
                        source = CodeSource("Patient")
                        code_usage[system][source][race_code] += 1
                    
                for ethnicity_entry in demographics.get("ethnicity_codings", []):
                    if isinstance(ethnicity_entry, dict) and ethnicity_entry.get("code"):
                        ethnicity_code = ethnicity_entry.get("code")
                        # Hard-coded system name
                        system = "ethnicity_code"
                        unique_codes[system].add(ethnicity_code)
                        # Don't track source field for hard-coded system
                        source = CodeSource("Patient")
                        code_usage[system][source][ethnicity_code] += 1
                
                # Process gender
                gender = demographics.get("gender")
                if gender:
                    # Hard-coded system name
                    system = "gender"
                    unique_codes[system].add(gender)
                    # Don't track source field for hard-coded system
                    source = CodeSource("Patient")
                    code_usage[system][source][gender] += 1
                    
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

def save_unique_codes(unique_codes: Dict[str, Set[str]], code_usage: Dict[str, Dict[CodeSource, Counter]], output_dir: str) -> None:
    """
    Save unique codes to CSV files, one file per code system.
    Also save code usage statistics including source field information.
    
    Args:
        unique_codes: Dictionary mapping code systems to sets of unique codes
        code_usage: Dictionary tracking code usage by resource type and source field
        output_dir: Directory to save the code files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each code system to a separate file
    for system, codes in unique_codes.items():
        # Create a safe filename from the original system
        safe_system_name = get_safe_filename(system)
        output_file = os.path.join(output_dir, f"{safe_system_name}.csv")
        
        # Save codes to CSV without unnecessary sorting
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["system", "code"])
            for code in codes:
                writer.writerow([system, code])
        
        logger.info(f"Saved {len(codes)} codes to {output_file}")
        
        # Save detailed usage for this code system
        if system in code_usage:
            usage_file = os.path.join(output_dir, f"{safe_system_name}_usage.csv")
            with open(usage_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Check if any source in this system has a source_field
                has_source_fields = any(source.source_field for source in code_usage[system].keys())
                
                if has_source_fields:
                    writer.writerow(["system", "code", "resource_type", "source_field", "count"])
                else:
                    writer.writerow(["system", "code", "resource_type", "count"])
                
                # For each code source where this code system appears
                for source, code_counts in code_usage[system].items():
                    # For each code and its count in this source
                    for code, count in code_counts.items():
                        if has_source_fields:
                            writer.writerow([system, code, source.resource_type, source.source_field, count])
                        else:
                            writer.writerow([system, code, source.resource_type, count])
            
            logger.info(f"Saved usage statistics to {usage_file}")
    
    # Create a summary file with counts
    summary_file = os.path.join(output_dir, "code_summary.csv")
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["system", "code_count", "resource_types_with_sources"])
        for system, codes in unique_codes.items():
            # Get the resource types and source fields where this code system appears
            sources = code_usage.get(system, {}).keys()
            
            # Create mapping of resource_type -> set of source_fields (add all resource types)
            resource_to_sources = defaultdict(set)
            for source in sources:
                # Always add the resource type to the dictionary, whether it has a source field or not
                if source.source_field:
                    resource_to_sources[source.resource_type].add(source.source_field)
                else:
                    # Add resource type with empty set of source fields
                    resource_to_sources[source.resource_type]
            
            # Format as ResourceType(source1,source2)|ResourceType2(source3)
            formatted_resources = []
            for resource_type, source_fields in resource_to_sources.items():
                if source_fields:  # If there are source fields, include them in parentheses
                    formatted_resources.append(f"{resource_type}({','.join(source_fields)})")
                else:  # If no source fields, just add the resource type
                    formatted_resources.append(resource_type)
                
            writer.writerow([system, len(codes), "|".join(formatted_resources)])
    
    logger.info(f"Saved summary to {summary_file}")
    
    # Create a detailed usage summary
    usage_summary_file = os.path.join(output_dir, "usage_summary.csv")
    with open(usage_summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["system", "resource_type", "source_field", "unique_codes", "total_occurrences"])
        
        for system, sources in code_usage.items():
            for source, code_counts in sources.items():
                resource_type = source.resource_type
                source_field = source.source_field if source.source_field else "N/A"
                unique_count = len(code_counts)
                total_occurrences = sum(code_counts.values())
                writer.writerow([system, resource_type, source_field, unique_count, total_occurrences])
    
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