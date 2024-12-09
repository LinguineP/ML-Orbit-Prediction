import os
import requests

# HTTP server details

# files are in the staging area only for 2 days after they are requested
base_url = "http://omniweb.gsfc.nasa.gov/staging/"
file_names = ["omni_min_vhXannIsX6.lst", "omni_min_vhXannIsX6.fmt"]

# Local directory to save the files
local_save_path = "./omnidata/downloaded_files"

# Ensure the local save directory exists
os.makedirs(local_save_path, exist_ok=True)


def download_files(base_url, file_names, save_path):
    for file_name in file_names:
        file_url = f"{base_url}{file_name}"
        local_file_path = os.path.join(save_path, file_name)
        try:
            print(f"Downloading {file_url}...")
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()  # Raise error for bad responses
            with open(local_file_path, "wb") as local_file:
                local_file.write(response.content)
            print(f"Downloaded: {file_name}")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")


# Run the download function
download_files(base_url, file_names, local_save_path)
