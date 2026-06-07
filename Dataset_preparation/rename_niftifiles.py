import os
import pandas as pd
import re
import shutil

def rename_nifti_based_on_csv(nifti_folder, csv_file_path):
    """
    Reads NIfTI files from a folder, filters them based on a CSV file
    for "Cognitively normal" subjects, and renames them by adding a prefix
    and age information.

    Args:
        nifti_folder (str): Path to the folder containing .nii.gz files.
        csv_file_path (str): Path to the CSV file with subject metadata.
    """

    # --- 1. Validate inputs ---
    if not os.path.isdir(nifti_folder):
        print(f"Error: NIfTI input folder '{nifti_folder}' not found.")
        return
    if not os.path.isfile(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return

    # --- 2. Read and filter CSV data ---
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file_path}': {e}")
        return

    # Filter for "Cognitively normal"
    cn_df = df[df['dx1'] == 'Cognitively normal'].copy()
    if cn_df.empty:
        print("No 'Cognitively normal' subjects found in the CSV file.")
        return

    # Create a dictionary for quick lookup: {csv_id: age}
    # Ensure age is float and handle potential NaNs robustly
    cn_data_map = {}
    for _, row in cn_df.iterrows():
        list_id = row['list1_id']
        try:
            age = float(row['ageAtVisit'])
            if pd.isna(age):
                print(f"Warning: Skipping CSV entry {list_id} due to missing age.")
                continue
            cn_data_map[list_id] = age
        except ValueError:
            print(f"Warning: Skipping CSV entry {list_id} due to invalid age format: {row['ageAtVisit']}.")
            continue
        
    if not cn_data_map:
        print("No valid 'Cognitively normal' subjects with age data found after processing CSV.")
        return
        
    print(f"Found {len(cn_data_map)} 'Cognitively normal' subjects with age in CSV.")

    # --- 3. Prepare output directory ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # if run in interactive environment where __file__ is not defined
        script_dir = os.getcwd()
        print(f"Warning: __file__ not defined. Using current working directory as script location: {script_dir}")

    input_folder_basename = os.path.basename(os.path.normpath(nifti_folder))
    if not input_folder_basename or input_folder_basename == ".":
        input_folder_basename = "input_nifti" # Fallback name

    output_parent_dir_name = "output" # Main output folder relative to script
    output_leaf_dir_name = f"{input_folder_basename}_renamed_CN"
    
    final_output_dir = os.path.join(script_dir, output_parent_dir_name, output_leaf_dir_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Renamed files will be saved in: {final_output_dir}\n")

    # --- 4. Process NIfTI files ---
    # Regex to parse NIfTI filenames
    # sub-OAS3<ID>_ses-d<SESSIONID>_run-<RUNID>_T1w_processed_.nii.gz
    # sub-OAS3<ID>_ses-d<SESSIONID>_T1w_processed_.nii.gz (no run)
    nifti_pattern = re.compile(
        r"sub-(OAS3\d+)_ses-(d\d+)(_run-(\d+))?_.*\.nii\.gz"
    )

    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(nifti_folder):
        if filename.endswith(".nii.gz"):
            match = nifti_pattern.match(filename)
            if match:
                subject_part = match.group(1)  # e.g., OAS30379
                session_part = match.group(2)  # e.g., d2106
                run_group = match.group(3)     # e.g., _run-02 (includes underscore and "run-")
                run_number_str = match.group(4) # e.g., 02 (just the number)

                # Construct the ID format as in the CSV's 'list1_id'
                # CSV format: OAS30001_MR_d0129
                csv_lookup_key = f"{subject_part}_MR_{session_part}"

                if csv_lookup_key in cn_data_map:
                    age = cn_data_map[csv_lookup_key]
                    age_str = f"{age:.2f}".replace('.', '_') # Format age like 48_87

                    # Construct new filename
                    # OASIS3_CN_OAS30009_MR_d0148_AGE_48.87_run-01_merged.nii.gz
                    new_filename_parts = [
                        "OASIS3_CN",
                        csv_lookup_key, # This is OAS3XXXX_MR_dYYYY
                        f"AGE_{age_str}"
                    ]
                    if run_number_str: # If run number exists
                        new_filename_parts.append(f"run-{run_number_str}")
                    
                    new_filename_parts.append("merged.nii.gz")
                    new_filename = "_".join(new_filename_parts)
                    
                    source_path = os.path.join(nifti_folder, filename)
                    destination_path = os.path.join(final_output_dir, new_filename)

                    try:
                        shutil.copy2(source_path, destination_path)
                        print(f"  Copied: '{filename}' -> '{new_filename}'")
                        processed_count += 1
                    except Exception as e:
                        print(f"  Error copying '{filename}' to '{new_filename}': {e}")
                        skipped_count += 1
                else:
                    # print(f"  Skipping '{filename}': Corresponding ID '{csv_lookup_key}' not found in filtered CSV data or not CN.")
                    skipped_count +=1
            else:
                print(f"  Skipping '{filename}': Does not match expected NIfTI filename pattern.")
                skipped_count +=1
        # else: not a .nii.gz file, ignore silently or add a log

    print(f"\n--- Summary ---")
    print(f"Total NIfTI files processed and renamed: {processed_count}")
    print(f"Total NIfTI files skipped (no match in CSV, pattern mismatch, or copy error): {skipped_count}")
    if processed_count > 0:
        print(f"Renamed files are located in: {final_output_dir}")

if __name__ == "__main__":
    nifti_input_folder = input("Enter the path to the folder with NIfTI files: ")
    csv_metadata_file = input("Enter the path to the CSV metadata file: ")

    rename_nifti_based_on_csv(nifti_input_folder, csv_metadata_file)
    print("\nProcessing complete.")