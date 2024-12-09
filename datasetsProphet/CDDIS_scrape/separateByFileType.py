import os
import shutil


def copy_files_by_type(source_dir, dest_dir):

    file_types = {
        ".dgf": "dgf_files",
        ".hts": "hts_files",
        ".sgf": "sgf_files",
        ".mcc": "mcc_files",
    }

    for dir_name in file_types.values():
        os.makedirs(os.path.join(dest_dir, dir_name), exist_ok=True)

    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if os.path.isfile(file_path):

            _, ext = os.path.splitext(file_name)
            if ext in file_types:
                dest_path = os.path.join(dest_dir, file_types[ext], file_name)
                shutil.copy(file_path, dest_path)
                print(f"Copied {file_name} to {file_types[ext]}")


source_directory = "D:\\fax\\master\\op-ml\\CDDIS_scrape\\CDDIS_larets_y2022"
destination_directory = (
    "D:\\fax\\master\\op-ml\\CDDIS_scrape\\CDDIS_larets_y2022_separated"
)
copy_files_by_type(source_directory, destination_directory)
