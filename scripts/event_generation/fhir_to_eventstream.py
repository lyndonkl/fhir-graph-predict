import os
import json
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from datetime import datetime, timezone, date

# Define the FHIR resource types we are interested in for events
TARGET_EVENT_RESOURCES = [
    "Condition", "Observation", "MedicationRequest",
    "Procedure", "Immunization", "DiagnosticReport"
]

# Define coding systems for specific codes
CODING_SYSTEMS = {
    "SNOMED": "http://snomed.info/sct",
    "LOINC": "http://loinc.org",
    "RXNORM": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "CVX": "http://hl7.org/fhir/sid/cvx",
    "CPT": "http://www.ama-assn.org/go/cpt", # Example, add others if needed
    "HCPCS": "https://bluebutton.cms.gov/resources/codesystem/hcpcs", # Example
    "UCUM": "http://unitsofmeasure.org"
}

# US Core Extension URLs
US_CORE_RACE_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
US_CORE_ETHNICITY_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"

def get_all_codings_from_concept(codeable_concept, target_system_url_or_prefixes: list):
    """Helper to extract all codes and their displays from a specific system (or list of systems) in a CodeableConcept."""
    if not codeable_concept or not isinstance(codeable_concept, dict) or 'coding' not in codeable_concept:
        return [] # Return empty list if no valid concept
    
    extracted_codings = []
    for coding_entry in codeable_concept.get('coding', []):
        if coding_entry and isinstance(coding_entry, dict) and 'system' in coding_entry and 'code' in coding_entry:
            # Check if the coding_entry's system matches any of the target_system_url_or_prefixes
            system_matched = False
            for target_system in target_system_url_or_prefixes:
                if coding_entry['system'].startswith(target_system):
                    system_matched = True
                    break
            
            if system_matched:
                extracted_codings.append({
                    "system": coding_entry['system'], # Store the actual system from the coding
                    "code": coding_entry['code'],
                    "display": coding_entry.get('display')
                })
    return extracted_codings

def get_value_from_observation(observation_resource):
    """Extracts value and unit from an Observation resource. Value can be a list of codings."""
    value_numeric = None
    value_concept_codings = [] # Will store a list of {"system":..., "code": ..., "display": ...}
    unit_code = None
    unit_display = None # This is the display name for the unit, not the UCUM code's display

    if 'valueQuantity' in observation_resource:
        value_numeric = observation_resource['valueQuantity'].get('value')
        unit_display = observation_resource['valueQuantity'].get('unit') # e.g., "mg/dL"
        unit_code = observation_resource['valueQuantity'].get('code') # UCUM code, e.g., "mg/dL"
        # We don't typically get a separate "display" for the UCUM code itself from valueQuantity's code.

    elif 'valueCodeableConcept' in observation_resource:
        # Prioritize SNOMED or LOINC for coded values
        value_concept_codings = get_all_codings_from_concept(
            observation_resource['valueCodeableConcept'],
            [CODING_SYSTEMS["SNOMED"], CODING_SYSTEMS["LOINC"]]
        )
        # Fallback to the first coding if no SNOMED/LOINC and codings exist
        if not value_concept_codings and observation_resource['valueCodeableConcept'].get('coding'):
            first_coding = observation_resource['valueCodeableConcept']['coding'][0]
            if first_coding and 'code' in first_coding: # Ensure basic structure
                value_concept_codings.append({
                    "system": first_coding.get('system', 'unknown_system'), # Add system if available
                    "code": first_coding.get('code'),
                    "display": first_coding.get('display')
                })
        # If text is available and no coding was processed, treat text as a code
        if not value_concept_codings and observation_resource['valueCodeableConcept'].get('text'):
             text_val = observation_resource['valueCodeableConcept'].get('text')
             value_concept_codings.append({"system": "text_value", "code": text_val, "display": text_val})

    elif 'valueString' in observation_resource:
        # Treat as a code/category for simplicity, display can be the same
        val_str = observation_resource.get('valueString')
        value_concept_codings.append({"system": "string_value", "code": val_str, "display": val_str})
    elif 'valueBoolean' in observation_resource:
        val_bool = str(observation_resource.get('valueBoolean'))
        value_concept_codings.append({"system": "boolean_value", "code": val_bool, "display": val_bool})
    # Add other value[x] types if necessary (e.g., valueDateTime, valuePeriod)

    return {
        "value_numeric": value_numeric,
        "value_concept_codings": value_concept_codings, # Note: pluralized
        "unit_code": unit_code, # UCUM code for the unit
        "unit_display": unit_display # Human-readable unit from valueQuantity.unit
    }

def get_interpretation_codings(observation_resource): # Renamed to reflect list return
    """Extracts interpretation codes and displays from an Observation resource."""
    if 'interpretation' in observation_resource and observation_resource['interpretation']:
        interpretation_concept_list = observation_resource['interpretation'] # interpretation is a list of CodeableConcepts
        
        all_interp_codings = []
        for interpretation_concept in interpretation_concept_list:
            # Try to get structured codes from the standard system for each concept
            codings = get_all_codings_from_concept(
                interpretation_concept,
                ["http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"] # Pass as list
            )
            if codings:
                all_interp_codings.extend(codings)
            # Fallback to text if no structured code found for this specific interpretation_concept
            elif interpretation_concept.get('text'):
                text = interpretation_concept.get('text')
                all_interp_codings.append({"system": "text_interpretation", "code": text, "display": text})
        
        if all_interp_codings:
            return all_interp_codings
            
    return [] # Return empty list if no interpretations


def get_event_timestamp(resource):
    """
    Extracts the most relevant timestamp for an event.
    Returns a datetime object if a parseable date is found, otherwise None.
    """
    timestamp_str = None
    if resource['resourceType'] == "Condition":
        timestamp_str = resource.get('onsetDateTime') or resource.get('recordedDate')
    elif resource['resourceType'] == "Observation":
        timestamp_str = resource.get('effectiveDateTime') or resource.get('issued')
    elif resource['resourceType'] == "MedicationRequest":
        timestamp_str = resource.get('authoredOn')
        # Could also consider encounter period if authoredOn is missing and MR is linked to an encounter
    elif resource['resourceType'] == "Procedure":
        timestamp_str = resource.get('performedDateTime')
        if not timestamp_str and 'performedPeriod' in resource:
            timestamp_str = resource['performedPeriod'].get('start')
    elif resource['resourceType'] == "Immunization":
        timestamp_str = resource.get('occurrenceDateTime')
    elif resource['resourceType'] == "DiagnosticReport":
        timestamp_str = resource.get('effectiveDateTime') or resource.get('issued')

    if timestamp_str:
        try:
            return parse_date(timestamp_str)
        except (ValueError, TypeError):
            print(f"Warning: Could not parse date '{timestamp_str}' for resource {resource.get('id', 'Unknown')}")
            return None
    return None


def extract_patient_demographics(patient_bundle):
    """Extracts patient demographics for the Patient node."""
    patient_resource = None
    for entry in patient_bundle.get('entry', []):
        resource = entry.get('resource', {})
        if resource.get('resourceType') == 'Patient':
            patient_resource = resource
            break

    if not patient_resource:
        return None

    demographics = {
        "fhir_patient_id": patient_resource.get('id'),
        "birthDate": patient_resource.get('birthDate'),
        "gender": patient_resource.get('gender'), # This is a simple string, not a coding structure
        "race_codings": [], # List of {"code": ..., "display": ...}
        "ethnicity_codings": [] # List of {"code": ..., "display": ...}
    }

    for ext in patient_resource.get('extension', []):
        if ext.get('url') == US_CORE_RACE_URL:
            for race_ext in ext.get('extension', []):
                if race_ext.get('url') == 'ombCategory' and 'valueCoding' in race_ext:
                    value_coding = race_ext['valueCoding']
                    demographics["race_codings"].append({
                        "code": value_coding.get('code'),
                        "display": value_coding.get('display')
                    })
        elif ext.get('url') == US_CORE_ETHNICITY_URL:
            for eth_ext in ext.get('extension', []):
                if eth_ext.get('url') == 'ombCategory' and 'valueCoding' in eth_ext:
                    value_coding = eth_ext['valueCoding']
                    demographics["ethnicity_codings"].append({
                        "code": value_coding.get('code'),
                        "display": value_coding.get('display')
                    })
    return demographics


def extract_clinical_events(patient_bundle):
    """Extracts all relevant clinical events with their properties."""
    processed_events = []
    patient_fhir_id = None

    for entry in patient_bundle.get('entry', []):
        resource = entry.get('resource', {})
        if resource.get('resourceType') == 'Patient':
            patient_fhir_id = resource.get('id')
            break

    for entry in patient_bundle.get('entry', []):
        resource = entry.get('resource', {})
        resource_type = resource.get('resourceType')

        if resource_type in TARGET_EVENT_RESOURCES:
            event_timestamp_dt = get_event_timestamp(resource)
            if not event_timestamp_dt:
                continue

            event_data = {
                "event_fhir_id": resource.get('id'),
                "patient_fhir_id": patient_fhir_id,
                "resourceType": resource_type,
                "event_timestamp": event_timestamp_dt.isoformat(),
                "year": event_timestamp_dt.year,
                "primary_codings": [], # Renamed from "codes"
                "status_codings": [],  # Renamed from "status_codes"
                "intent_codings": [],  # Renamed from "intent_codes"
                "interpretation_codings": [], # Renamed from "interpretation_codes"
                "value_numeric": None,
                "value_concept_codings": [], # Now a list
                "unit_code": None, 
                "unit_display": None # Human-readable unit string
            }

            if resource_type == "Condition":
                codings = get_all_codings_from_concept(resource.get('code'), [CODING_SYSTEMS["SNOMED"]])
                if codings: event_data["primary_codings"].extend(codings)

                clinical_status_codings = get_all_codings_from_concept(resource.get('clinicalStatus'), ["http://terminology.hl7.org/CodeSystem/condition-clinical"])
                if clinical_status_codings: event_data["status_codings"].extend(clinical_status_codings)

                verification_status_codings = get_all_codings_from_concept(resource.get('verificationStatus'), ["http://terminology.hl7.org/CodeSystem/condition-ver-status"]) # Corrected URL from previous attempt
                if verification_status_codings: event_data["status_codings"].extend(verification_status_codings)

            elif resource_type == "Observation":
                codings = get_all_codings_from_concept(resource.get('code'), [CODING_SYSTEMS["LOINC"]])
                if codings: event_data["primary_codings"].extend(codings)

                value_info = get_value_from_observation(resource)
                event_data["value_numeric"] = value_info.get("value_numeric")
                # value_info["value_concept_codings"] is already a list
                if value_info.get("value_concept_codings"):
                    event_data["value_concept_codings"].extend(value_info["value_concept_codings"])
                event_data["unit_code"] = value_info.get("unit_code")
                event_data["unit_display"] = value_info.get("unit_display")

                interp_codings_list = get_interpretation_codings(resource) # Returns a list
                if interp_codings_list: 
                    event_data["interpretation_codings"].extend(interp_codings_list)

            elif resource_type == "MedicationRequest":
                codings = get_all_codings_from_concept(resource.get('medicationCodeableConcept'), [CODING_SYSTEMS["RXNORM"]])
                if codings: event_data["primary_codings"].extend(codings)
                
                status = resource.get('status')
                # Statuses are single string values, not CodeableConcepts, so append as single structured dict to the list
                if status: event_data["status_codings"].append({"system": "medicationrequest-status", "code": status, "display": status})

                intent = resource.get('intent')
                if intent: event_data["intent_codings"].append({"system": "medicationrequest-intent", "code": intent, "display": intent}) # intent_codings is a list

            elif resource_type == "Procedure":
                codings = get_all_codings_from_concept(resource.get('code'), [CODING_SYSTEMS["SNOMED"], CODING_SYSTEMS["CPT"], CODING_SYSTEMS["HCPCS"]])
                if codings: event_data["primary_codings"].extend(codings)

                status = resource.get('status')
                if status: event_data["status_codings"].append({"system": "procedure-status", "code": status, "display": status})

            elif resource_type == "Immunization":
                codings = get_all_codings_from_concept(resource.get('vaccineCode'), [CODING_SYSTEMS["CVX"]])
                if codings: event_data["primary_codings"].extend(codings)

                status = resource.get('status')
                if status: event_data["status_codings"].append({"system": "immunization-status", "code": status, "display": status})

            elif resource_type == "DiagnosticReport":
                # Report code, often LOINC
                codings = get_all_codings_from_concept(resource.get('code'), [CODING_SYSTEMS["LOINC"]])
                if codings: event_data["primary_codings"].extend(codings) 

                status = resource.get('status')
                if status: event_data["status_codings"].append({"system": "diagnosticreport-status", "code": status, "display": status})

            processed_events.append(event_data)

    # Sort events by timestamp for chronological order
    processed_events.sort(key=lambda x: x['event_timestamp'])
    return processed_events


def calculate_age_at_event(birth_date_str, event_date_dt):
    """Calculates age at the time of an event."""
    if not birth_date_str or not event_date_dt:
        return None
    try:
        # Handle cases where birth_date_str might only be YYYY or YYYY-MM
        if len(birth_date_str) == 4: # YYYY
            birth_date = date(int(birth_date_str), 1, 1)
        elif len(birth_date_str) == 7: # YYYY-MM
             year, month = map(int, birth_date_str.split('-'))
             birth_date = date(year, month, 1)
        else: # Assume YYYY-MM-DD
            birth_date = parse_date(birth_date_str).date()

        # Ensure event_date_dt is just a date object for comparison
        event_date_only = event_date_dt.date() if isinstance(event_date_dt, datetime) else event_date_dt
        
        age = event_date_only.year - birth_date.year - \
              ((event_date_only.month, event_date_only.day) < (birth_date.month, birth_date.day))
        return age
    except ValueError:
        print(f"Warning: Could not parse birth date '{birth_date_str}' for age calculation.")
        return None

def process_patient_fhir_bundle(patient_fhir_path, output_dir):
    """Processes a single patient's FHIR bundle file."""
    try:
        with open(patient_fhir_path, 'r') as infile:
            patient_bundle = json.load(infile)
    except Exception as e:
        print(f"Error reading or parsing JSON file {patient_fhir_path}: {e}")
        return

    demographics = extract_patient_demographics(patient_bundle)
    if not demographics:
        print(f"Could not extract demographics from {patient_fhir_path}")
        return

    clinical_events = extract_clinical_events(patient_bundle)

    # Add age_at_event to each clinical event
    if demographics.get("birthDate"):
        for event in clinical_events:
            event_dt = parse_date(event["event_timestamp"])
            event["age_at_event"] = calculate_age_at_event(demographics["birthDate"], event_dt)

    patient_id_from_filename = os.path.basename(patient_fhir_path).split('.')[0] # Assuming filename is patient_id.json

    patient_output_data = {
        "patient_id_from_filename": patient_id_from_filename, # For reference
        "demographics": demographics, # Contains fhir_patient_id
        "clinical_events": clinical_events # Contains event_timestamp, year, fhir_event_id etc.
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{patient_id_from_filename}_processed.json')
    try:
        with open(output_path, 'w') as outfile:
            json.dump(patient_output_data, outfile, indent=2, default=str) # Use default=str for datetime objects if not already isoformat
        # print(f"Successfully processed and saved: {output_path}")
    except Exception as e:
        print(f"Error writing JSON output for {patient_id_from_filename}: {e}")


if __name__ == "__main__":
    # These paths will need to be adjusted to your actual directory structure
    # Assuming your FHIR files are in a directory named 'fhir_data'
    # and you want the output in 'processed_data'
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the script

    # Example: If your script is in a 'scripts' folder and data is in a parallel 'data/fhir_input'
    # fhir_input_directory = os.path.join(base_dir, '..', 'data', 'fhir_input')
    # processed_output_directory = os.path.join(base_dir, '..', 'data', 'processed_ehr_data')

    # --- USER: PLEASE ADJUST THESE PATHS ---
    # For testing with one of the uploaded files, you'd point fhir_input_directory to where that file is.
    # Let's assume you have downloaded 'Abraham100_Bogisich202_ba6abbf2-2aae-b7af-604a-02913a3995f1.json'
    # into a folder named 'sample_fhir_files' next to your script.
    
    # For a single file test:
    # current_script_dir = os.path.dirname(__file__) # Directory of the script
    # single_fhir_file_path = os.path.join(current_script_dir, "sample_fhir_files", "Abraham100_Bogisich202_ba6abbf2-2aae-b7af-604a-02913a3995f1.json") # ADJUST FILENAME
    # processed_output_directory = os.path.join(current_script_dir, "processed_ehr_data")
    # if os.path.exists(single_fhir_file_path):
    #     print(f"Processing single file: {single_fhir_file_path}")
    #     process_patient_fhir_bundle(single_fhir_file_path, processed_output_directory)
    # else:
    #     print(f"Test file not found: {single_fhir_file_path}")

    # For processing a directory of FHIR files:
    fhir_input_directory = "../../../synthea/output/fhir"  # ADJUST THIS to your FHIR files directory
    processed_output_directory = "../../data/processed_ehr_data" # ADJUST THIS to your desired output directory

    if not os.path.isdir(fhir_input_directory):
        print(f"Error: FHIR input directory not found at {fhir_input_directory}")
        print("Please adjust the 'fhir_input_directory' variable in the script.")
    else:
        patient_fhir_files = [f for f in os.listdir(fhir_input_directory) if f.endswith('.json')]
        if not patient_fhir_files:
            print(f"No JSON files found in {fhir_input_directory}")
        else:
            print(f"Found {len(patient_fhir_files)} FHIR files to process from {fhir_input_directory}")
            for fname in tqdm(patient_fhir_files, desc="Processing patient files"):
                file_path = os.path.join(fhir_input_directory, fname)
                process_patient_fhir_bundle(file_path, processed_output_directory)
            print(f"Processing complete. Output saved to {processed_output_directory}")