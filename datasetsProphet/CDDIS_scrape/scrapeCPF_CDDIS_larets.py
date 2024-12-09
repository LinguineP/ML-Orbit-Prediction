import requests
import os

DOWNLOAD_PATH = "D:\\fax\\master\\op-ml\\CDDIS_scrape\\CDDIS_larets_y2022"  # Define the global constant for the download path
BASE_URL = "https://cddis.nasa.gov/archive/slr/cpf_predicts/2022/larets/"  # Update this to the base URL


def download_file(file_url):
    """Downloads a file from the given URL and saves it to the globally defined directory."""
    # Ensure the file_url is complete by prepending the base URL
    if not file_url.startswith("https"):
        file_url = BASE_URL + file_url.strip()  # Prepend the base URL to the file URL

    # Extract the file name from the URL (last part of the URL)
    filename = file_url.split("/")[-1]

    # Ensure the directory exists
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    # Full path to save the file
    file_path = os.path.join(DOWNLOAD_PATH, filename)

    # Makes a request to download the file
    file_r = requests.get(file_url)

    # Saves the file locally to the specified path
    with open(file_path, "wb") as fd:
        for chunk in file_r.iter_content(chunk_size=1000):
            fd.write(chunk)

    print(f"Downloaded: {filename} to {file_path}")


def list_and_download_files(url):
    """Lists the files from the directory and downloads each one to the globally defined path."""
    # Adds '*?list' to the end of URL if not included already to list directory contents
    if not url.endswith("*?list"):
        url = url + "*?list"

    # Makes a request to the URL, stores the response in variable r
    r = requests.get(url)

    # Prints the directory listing
    print("Directory Listing:")
    print(r.text)

    for line in r.text.splitlines():
        filename = line.split()[0]  # Get only the filename part
        if filename.endswith(".sgf"):  # Check if it matches the desired extension
            download_file(filename)
        elif filename.endswith(".hts"):
            download_file(filename)
        elif filename.endswith(".dgf"):
            download_file(filename)
        elif filename.endswith(".mcc"):
            download_file(filename)


if __name__ == "__main__":
    """requires .netrc see more here:  https://cddis.nasa.gov/Data_and_Derived_Products/CDDIS_Archive_Access.html"""

    # Reads the URL from the command line argument
    url = "https://cddis.nasa.gov/archive/slr/cpf_predicts/2022/larets/"

    list_and_download_files(url)
