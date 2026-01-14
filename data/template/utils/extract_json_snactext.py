import argparse
import os
import json
from tqdm import tqdm

def append_to_file(file_path, field, snac):
    """Append text data to file incrementally."""
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write('#B: ' + snac + '\n')
            file.write('#U: ' + field + '\n')
    else:
        with open(file_path, 'w') as file:
            file.write('#B: ' + snac + '\n')
            file.write('#U: ' + field + '\n')


def process_text_extraction(input_path, value, output_path):
    """Process the text file specified and extract text and write it to a file."""
    with open(input_path, 'r') as file:
        data = json.load(file)

    for entry in data:
        field = ""
        value = value.lower()
        field = entry.get(value, "")
        raw_snac = entry["sequential_snac_tokens"]
        snac = ""

        for i in raw_snac:
            num = f"<snac>{i}</snac>"
            snac = snac + num
        
        append_to_file(output_path, field, snac)


def process_directory(input_path, value, output_path):
    """Process specified directory."""
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    for filename in tqdm(json_files, desc="Processing json files"):
        file_path = os.path.join(input_path, filename)
        process_text_extraction(file_path, value, output_path)


def main():
    parser = argparse.ArgumentParser(description="Extracting text data from snac_json output json file.")
    parser.add_argument("input", help="Input json file path to extract text data from.")
    parser.add_argument("value", help="Field to extract value from.")
    parser.add_argument("output", help="Output path to store the result file.")
    parser.add_argument('--directory', action='store_true', help="Process all json file in the input directory.")
    args = parser.parse_args()

    if args.directory:
        process_directory(args.input, args.value, args.output)
    else:
        process_text_extraction(args.input, args.value, args.output)

if __name__ == "__main__":
    main()
