import io
import zipfile

import requests

DATASET_URL = 'https://data.open-power-system-data.org/time_series/opsd-time_series-2020-10-06.zip'
OUTPUT_PATH = ''


def load_dataset_as_zip(dataset_url: str, output_path: str):
    r = requests.get(dataset_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(output_path)


def main():
    load_dataset_as_zip(DATASET_URL, OUTPUT_PATH)


if __name__ == '__main__':
    main()
