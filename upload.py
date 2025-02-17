import streamlit as st
import pandas as pd
import os
import random
import shutil


def save_files(uploaded_files, base_path, folder_id):
    """Save uploaded .dat and .hea files into the folder for the given folder_id.
       Overwrites any existing folder."""
    # Folder to store the files
    new_dir = os.path.join(base_path, str(folder_id))
    # If folder exists, remove it entirely
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.dat') or uploaded_file.name.endswith(
                '.hea'):
            ext = uploaded_file.name.split('.')[-1]
            new_file_name = f"{folder_id}.{ext}"
            with open(os.path.join(new_dir, new_file_name), "wb") as f:
                f.write(uploaded_file.getbuffer())


def main():
    st.title("ECG Data Upload")

    # File uploader for .dat and .hea files
    uploaded_files = st.file_uploader("Upload .dat and .hea files",
                                      type=['dat', 'hea'],
                                      accept_multiple_files=True)

    # Gather user input using Streamlit's form
    with st.form(key='input_form'):
        label = st.text_input("Enter label (e.g., normal, mi, nstemi):")
        age = st.number_input("Enter age:",
                              min_value=0,
                              max_value=120,
                              value=30)
        sex = st.selectbox("Select sex:",
                           options=[0, 1],
                           format_func=lambda x: "Male"
                           if x == 0 else "Female")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if uploaded_files:
            # Set base paths (adjust these paths as necessary)
            base_path = os.path.join(os.getcwd(), "data", "Test_data")
            metadata_file = os.path.join(os.getcwd(), "data", "Test_metadata.csv")


            # We want only one entry. Use folder id 1 always.
            folder_id = 1

            # Save the uploaded files into the designated folder (overwriting old files)
            save_files(uploaded_files, base_path, folder_id)

            # Generate a unique patient id (new, since we're overwriting previous data)
            patient_id = random.randint(10000, 99999)

            # Construct the new record as a DataFrame.
            # Here, we assume that the 'path' field is set relative to your data folder.
            new_record = pd.DataFrame({
                "id_rnd": [folder_id],
                "label": [label],
                "patient_id": [patient_id],
                "path": [f"/{folder_id}/{folder_id}"],
                "age": [age],
                "sex": [sex]
            })

            # Overwrite any existing metadata file with the new record
            new_record.to_csv(metadata_file, index=False)

            st.success("Files uploaded and metadata saved successfully!")
        else:
            st.warning("Please upload at least one .dat or .hea file.")


if __name__ == "__main__":
    main()
