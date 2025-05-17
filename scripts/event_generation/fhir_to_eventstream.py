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

def get_first_coding_with_system(codeable_concept, target_system_url_or_prefix):
    """Helper to extract the first code from a specific system in a CodeableConcept."""
    if not codeable_concept or not isinstance(codeable_concept, dict) or 'coding' not in codeable_concept:
        return None
    for coding in codeable_concept.get('coding', []):
        if coding and isinstance(coding, dict) and 'system' in coding and 'code' in coding:
            if isinstance(target_system_url_or_prefix, list): # If multiple system aliases
                 for prefix in target_system_url_or_prefix:
                    if coding['system'].startswith(prefix):
                        return coding['code']
            elif coding['system'].startswith(target_system_url_or_prefix): # if single system alias
                return coding['code']
    return None # Or return all codings if specific system not found / prefer more generic one

def get_value_from_observation(observation_resource):
    """Extracts value and unit from an Observation resource."""
    value_numeric = None
    value_code = None
    unit_code = None
    unit_display = None

    if 'valueQuantity' in observation_resource:
        value_numeric = observation_resource['valueQuantity'].get('value')
        unit_display = observation_resource['valueQuantity'].get('unit')
        unit_code = observation_resource['valueQuantity'].get('code') # UCUM code

    elif 'valueCodeableConcept' in observation_resource:
        # For now, we'll prioritize SNOMED or LOINC for coded values if available,
        # otherwise, take the first available code.
        value_code = get_first_coding_with_system(observation_resource['valueCodeableConcept'],
                                                [CODING_SYSTEMS["SNOMED"], CODING_SYSTEMS["LOINC"]])
        if not value_code and observation_resource['valueCodeableConcept'].get('coding'):
             value_code = observation_resource['valueCodeableConcept']['coding'][0].get('code')


    elif 'valueString' in observation_resource:
        value_code = observation_resource.get('valueString') # Treat as a code/category for simplicity
    elif 'valueBoolean' in observation_resource:
        value_code = str(observation_resource.get('valueBoolean'))
    # Add other value[x] types if necessary (e.g., valueDateTime, valuePeriod)

    return {
        "value_numeric": value_numeric,
        "value_codeable_concept_code": value_code, # This can be a direct code or a string
        "unit_code": unit_code, # UCUM
        "unit_display": unit_display
    }

def get_interpretation_code(observation_resource):
    """Extracts interpretation code from an Observation resource."""
    if 'interpretation' in observation_resource and observation_resource['interpretation']:
        # Typically, interpretation is a list, but we'll take the first one for simplicity
        interpretation_concept = observation_resource['interpretation'][0]
        # Prioritize a standard system if available, otherwise take text or first code
        # Example systems for interpretation: http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation
        return get_first_coding_with_system(interpretation_concept, "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation") or \
               interpretation_concept.get('text')
    return None


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
        "fhir_patient_id": patient_resource.get('id'), # For later linking
        "birthDate": patient_resource.get('birthDate'), # String, will be used for age calculation
        "gender": patient_resource.get('gender'), # String
        "race_codes": [], # List of OMB race codes
        "ethnicity_codes": [] # List of OMB ethnicity codes
    }

    for ext in patient_resource.get('extension', []):
        if ext.get('url') == US_CORE_RACE_URL:
            for race_ext in ext.get('extension', []):
                if race_ext.get('url') == 'ombCategory' and 'valueCoding' in race_ext:
                    demographics["race_codes"].append(race_ext['valueCoding'].get('code'))
        elif ext.get('url') == US_CORE_ETHNICITY_URL:
            for eth_ext in ext.get('extension', []):
                if eth_ext.get('url') == 'ombCategory' and 'valueCoding' in eth_ext:
                    demographics["ethnicity_codes"].append(eth_ext['valueCoding'].get('code'))
    return demographics


def extract_clinical_events(patient_bundle):
    """Extracts all relevant clinical events with their properties."""
    processed_events = []
    patient_fhir_id = None # To associate events with the patient

    # First, find the patient ID
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
                # print(f"Skipping resource {resource.get('id')} due to missing timestamp.")
                continue # Skip if no valid timestamp for the event itself

            event_data = {
                "event_fhir_id": resource.get('id'), # For later linking/debugging
                "patient_fhir_id": patient_fhir_id,
                "resourceType": resource_type,
                "event_timestamp": event_timestamp_dt.isoformat(), # Standard ISO format
                "year": event_timestamp_dt.year, # For assigning to AnnualSnapshot
                # Specific fields for each resource type
                "codes": [], # To store primary codes (SNOMED, LOINC, RXNORM etc.)
                "status_codes": [],
                "intent_codes": [], # For MedicationRequest
                "interpretation_codes": [], # For Observation
                "value_numeric": None,
                "value_codeable_concept_code": None,
                "unit_code": None, # UCUM
                "unit_display": None
            }

            if resource_type == "Condition":
                snomed_code = get_first_coding_with_system(resource.get('code'), CODING_SYSTEMS["SNOMED"])
                if snomed_code: event_data["codes"].append({"system": "SNOMED", "code": snomed_code})

                clinical_status_code = get_first_coding_with_system(resource.get('clinicalStatus'), "http://terminology.hl7.org/CodeSystem/condition-clinical")
                if clinical_status_code: event_data["status_codes"].append({"system": "condition-clinical", "code": clinical_status_code})

                verification_status_code = get_first_coding_with_system(resource.get('verificationStatus'), "http://terminology.hl7.org/CodeSystem/condition-verification")
                if verification_status_code: event_data["status_codes"].append({"system": "condition-verification", "code": verification_status_code})


            elif resource_type == "Observation":
                loinc_code = get_first_coding_with_system(resource.get('code'), CODING_SYSTEMS["LOINC"])
                if loinc_code: event_data["codes"].append({"system": "LOINC", "code": loinc_code})

                value_info = get_value_from_observation(resource)
                event_data.update(value_info) # Adds value_numeric, value_codeable_concept_code, unit_code, unit_display

                interp_code = get_interpretation_code(resource)
                if interp_code: event_data["interpretation_codes"].append(interp_code) # Could be a code or text

            elif resource_type == "MedicationRequest":
                rxnorm_code = get_first_coding_with_system(resource.get('medicationCodeableConcept'), CODING_SYSTEMS["RXNORM"])
                if rxnorm_code: event_data["codes"].append({"system": "RXNORM", "code": rxnorm_code})
                # Could also check resource.get('medicationReference') if it's a reference

                status = resource.get('status')
                if status: event_data["status_codes"].append({"system": "medicationrequest-status", "code": status}) # FHIR uses string values directly for status

                intent = resource.get('intent')
                if intent: event_data["intent_codes"].append({"system": "medicationrequest-intent", "code": intent}) # FHIR uses string values directly for intent

            elif resource_type == "Procedure":
                # Procedures can have SNOMED, CPT, HCPCS etc. Prioritize SNOMED for now.
                proc_code = get_first_coding_with_system(resource.get('code'), [CODING_SYSTEMS["SNOMED"], CODING_SYSTEMS["CPT"], CODING_SYSTEMS["HCPCS"]])
                # You might want to store the system along with the code if it's not always SNOMED
                if proc_code: event_data["codes"].append({"system": "ProcedureCode", "code": proc_code}) # Generic system name for now

                status = resource.get('status')
                if status: event_data["status_codes"].append({"system": "procedure-status", "code": status})

            elif resource_type == "Immunization":
                cvx_code = get_first_coding_with_system(resource.get('vaccineCode'), CODING_SYSTEMS["CVX"])
                if cvx_code: event_data["codes"].append({"system": "CVX", "code": cvx_code})

                status = resource.get('status')
                if status: event_data["status_codes"].append({"system": "immunization-status", "code": status})

            elif resource_type == "DiagnosticReport":
                # Report code, often LOINC
                report_code = get_first_coding_with_system(resource.get('code'), CODING_SYSTEMS["LOINC"])
                if report_code: event_data["codes"].append({"system": "LOINC", "code": report_code}) # For report type

                status = resource.get('status')
                if status: event_data["status_codes"].append({"system": "diagnosticreport-status", "code": status})

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