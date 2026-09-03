import argparse
import os
import re
from tqdm import tqdm

def append_to_file(file_path, snac):
    """Append text data to file incrementally."""
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(snac + '\n')
    else:
        with open(file_path, 'w') as file:
            file.write(snac + '\n')


def process_text_extraction(input_path, output_path):
    """Process the text file specified and extract snac and write it to a file."""
    with open(input_path, 'r') as file:
        data = file.read()

    pattern = r'<snac>(\d+)</snac>'
    matches = re.findall(pattern, data)
    for match in matches:
        append_to_file(output_path, match)


def process_directory(input_path, output_path):
    """Process specified directory."""
    json_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
    for filename in tqdm(json_files, desc="Processing txt files"):
        file_path = os.path.join(input_path, filename)
        process_text_extraction(file_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Extracting snac data from snac sample output.")
    parser.add_argument("input", help="Input text file path to extract snac data from.")
    parser.add_argument("output", help="Output path to store the extracted snac data file.")
    parser.add_argument('--directory', action='store_true', help="Process all json file in the input directory.")
    args = parser.parse_args()

    if args.directory:
        process_directory(args.input, args.output)
    else:
        process_text_extraction(args.input, args.output)

if __name__ == "__main__":
    main()
