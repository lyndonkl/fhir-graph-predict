#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate and plot the age distribution of patients at their last event.
This script processes patient data, calculates ages, bins them,
and saves a histogram plot and CSV of the distribution.
"""

import os
import json
import logging
import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter
from datetime import datetime, date

from dateutil.parser import parse as parse_date
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def calculate_age(birth_date_str: Optional[str], reference_event_date_str: Optional[str]) -> Optional[int]:
    """
    Calculates age based on birth date and a reference event date.

    Args:
        birth_date_str: The patient's birth date as a string (YYYY, YYYY-MM, or YYYY-MM-DD).
        reference_event_date_str: The reference event date as an ISO format string.

    Returns:
        The calculated age as an integer, or None if calculation is not possible.
    """
    if not birth_date_str or not reference_event_date_str:
        return None

    try:
        reference_date_dt = parse_date(reference_event_date_str)
        
        # Handle different birth date string formats
        if len(birth_date_str) == 4:  # YYYY
            birth_date = date(int(birth_date_str), 1, 1)
        elif len(birth_date_str) == 7:  # YYYY-MM
            year, month = map(int, birth_date_str.split('-'))
            birth_date = date(year, month, 1)
        else:  # Assume YYYY-MM-DD or other parseable full date
            birth_date = parse_date(birth_date_str).date()

        # Ensure reference_date_dt is just a date object for comparison
        reference_date_only = reference_date_dt.date()
        
        age = reference_date_only.year - birth_date.year - \
              ((reference_date_only.month, reference_date_only.day) < (birth_date.month, birth_date.day))
        return age
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse dates for age calculation. Birth: '{birth_date_str}', Event: '{reference_event_date_str}'. Error: {e}")
        return None

def get_patient_final_ages(data_dir: str) -> List[int]:
    """
    Scans through patient data files to determine the age of each patient at their last event.

    Args:
        data_dir: Directory containing processed EHR data files (JSON format).

    Returns:
        A list of ages (integers).
    """
    patient_final_ages: List[int] = []
    json_files = list(Path(data_dir).glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}. Cannot calculate age distribution.")
        return []

    logger.info(f"Found {len(json_files)} JSON files in {data_dir} for age calculation.")
    
    for file_path in tqdm(json_files, desc="Calculating patient ages"):
        try:
            with open(file_path, "r") as f:
                patient_data = json.load(f)
            
            demographics = patient_data.get("demographics")
            clinical_events = patient_data.get("clinical_events", [])
            
            if not demographics or not clinical_events:
                logger.warning(f"Skipping {file_path}: Missing demographics or clinical events.")
                continue
                
            birth_date_str = demographics.get("birthDate")
            if not birth_date_str:
                logger.warning(f"Skipping patient in {file_path}: Missing birthDate.")
                continue

            # Clinical events are assumed to be sorted chronologically.
            # The last event is the final one in the list.
            last_event = clinical_events[-1]
            last_event_timestamp_str = last_event.get("event_timestamp")

            if not last_event_timestamp_str:
                logger.warning(f"Skipping patient in {file_path}: Last event has no timestamp.")
                continue
                
            age = calculate_age(birth_date_str, last_event_timestamp_str)
            if age is not None and age >= 0: # Ensure age is non-negative
                patient_final_ages.append(age)
            elif age is not None:
                logger.warning(f"Calculated negative age for patient in {file_path}. Birth: {birth_date_str}, Last Event: {last_event_timestamp_str}. Age: {age}. Skipping.")

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {file_path}")
        except IndexError:
            logger.warning(f"Skipping {file_path}: No clinical events found after loading.")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
    logger.info(f"Successfully calculated final ages for {len(patient_final_ages)} patients.")
    return patient_final_ages

def generate_and_save_distribution(
    ages: List[int], 
    output_dir: str, 
    bin_width: int = 5
) -> None:
    """
    Generates an age distribution histogram and saves it as a PNG file.
    Also saves the binned distribution data to a CSV file.

    Args:
        ages: A list of patient ages.
        output_dir: Directory to save the plot and CSV file.
        bin_width: The width of each age bin (e.g., 5 for 5-year increments).
    """
    if not ages:
        logger.warning("No ages provided. Skipping plot generation and CSV saving.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Determine min and max age for bins
    min_age = 0 # Start bins from 0
    max_age = max(ages) if ages else 0
    
    # Create bins: e.g., [0, 5, 10, ..., max_age_rounded_up_to_nearest_bin_width]
    bins = np.arange(min_age, max_age + bin_width, bin_width)
    
    # Generate histogram data
    counts, bin_edges = np.histogram(ages, bins=bins)
    
    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    plt.bar(bin_edges[:-1], counts, width=bin_width * 0.9, align='edge', edgecolor='black')
    
    plt.xlabel(f"Age Group ({bin_width}-Year Increments)")
    plt.ylabel("Number of Patients")
    plt.title("Patient Age Distribution at Last Clinical Event")
    
    # Create meaningful x-tick labels (e.g., "0-4", "5-9")
    tick_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}" for i in range(len(bin_edges)-1)]
    plt.xticks(ticks=bin_edges[:-1], labels=tick_labels, rotation=45, ha="right")
    
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    plot_path = os.path.join(output_dir, "age_distribution.png")
    try:
        plt.savefig(plot_path)
        logger.info(f"Age distribution plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    plt.close()

    # --- Save CSV ---
    csv_path = os.path.join(output_dir, "age_distribution.csv")
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["AgeGroup", "Count"])
            for i in range(len(counts)):
                age_group_label = f"{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}"
                writer.writerow([age_group_label, counts[i]])
        logger.info(f"Age distribution data saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save age distribution CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description="Calculate and plot patient age distribution.")
    parser.add_argument(
        "--data_dir", 
        default="data/processed_ehr_data",
        help="Directory containing processed EHR data files (JSON format). Default: data/processed_ehr_data"
    )
    parser.add_argument(
        "--output_dir", 
        default="data/plots",
        help="Directory to save the age distribution plot and CSV. Default: data/plots"
    )
    parser.add_argument(
        "--bin_width",
        type=int,
        default=5,
        help="Width of age bins for the histogram (in years). Default: 5"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting age distribution calculation from {args.data_dir}.")
    logger.info(f"Output will be saved to {args.output_dir} with bin width {args.bin_width}.")
    
    patient_ages = get_patient_final_ages(args.data_dir)
    
    if patient_ages:
        generate_and_save_distribution(patient_ages, args.output_dir, args.bin_width)
    else:
        logger.info("No patient ages collected. Exiting.")
        
    logger.info("Age distribution script finished.")

if __name__ == "__main__":
    main() 