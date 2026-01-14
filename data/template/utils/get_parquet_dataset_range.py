import pandas as pd
import requests
import os
import argparse
import json
from tqdm import tqdm
# Note: BeautifulSoup import removed as it's no longer needed

def download_file(url, filename):
    """
    Download a file from a given URL with a progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the download was successful.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
        else:
            progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Failed to download the file completely.")
    else:
        print(f"Downloaded {filename}")


def convert_to_json(parquet_path, json_path):
    """
    Convert Parquet file to JSON.
    """
    if not os.path.exists(json_path):
        df = pd.read_parquet(parquet_path)
        df.to_json(json_path, orient="records")
        print(f"Converted {parquet_path} to JSON at {json_path}")
    else:
        print(f"{json_path} already exists, continuing")


def emit_json_contents(
    json_path,
    output_text_file,
    include_keys,
    value_prefixes,
    required_key,
    skip_empty,
    exclude,
    list_key=None,
    role_prefixes=None,
):
    """
    Emit the contents of the JSON file.
    Optionally, write the output to a text file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    excluded_pairs = {}
    if exclude:
        for pair in exclude:
            for i in range(0, len(pair), 2):
                key = pair[i]
                value = pair[i + 1]
                excluded_pairs[key] = value

    # Build role_prefixes dict if provided
    role_prefix_dict = {}
    if role_prefixes:
        if len(role_prefixes) % 2 != 0:
            raise ValueError(
                "role_prefixes must contain pairs of ROLE and PREFIX. Please check your input."
            )
        for i in range(0, len(role_prefixes), 2):
            role = role_prefixes[i]
            prefix = role_prefixes[i + 1]
            role_prefix_dict[role] = prefix

    with open(output_text_file, "a") as f:
        prev_item_written = False
        for item in data:
            if required_key and item.get(required_key, "") == "":
                continue  # Skip entire item if required key is empty

            skip_item = False
            # Apply excludes
            for ex_key, ex_value in excluded_pairs.items():
                if item.get(ex_key) == ex_value:
                    skip_item = True
                    break
            if skip_item:
                continue

            if list_key and list_key in item and isinstance(item[list_key], list):
                # Process sub_items in the order they appear
                sub_items = item[list_key]
                for sub_item in sub_items:
                    role = sub_item.get('role')
                    if role_prefix_dict and role not in role_prefix_dict:
                        continue  # Skip roles not in role_prefixes
                    skip_sub_item = False

                    # Apply excludes to sub_item
                    for ex_key, ex_value in excluded_pairs.items():
                        if sub_item.get(ex_key) == ex_value:
                            skip_sub_item = True
                            break
                    if skip_sub_item:
                        continue

                    if required_key and sub_item.get(required_key, "") == "":
                        continue  # Skip if required key is empty in sub_item

                    # Get prefix
                    prefix = role_prefix_dict.get(role, "") if role_prefix_dict else ""

                    # Build content line
                    content_pieces = []
                    for key in include_keys:
                        if key in sub_item:
                            value = sub_item[key]
                            if skip_empty and value == "":
                                continue
                            content_pieces.append(value)

                    if content_pieces:
                        content_line = prefix + ' '.join(content_pieces)
                        if prev_item_written:
                            f.write("\n")  # Single newline between items
                        f.write(content_line.strip())
                        prev_item_written = True
            else:
                # Process item directly
                content_written = False
                for key, prefix in zip(include_keys, value_prefixes):
                    if key in item:
                        value = item[key]
                        if skip_empty and value == "":
                            continue
                        if prev_item_written or content_written:
                            f.write("\n")  # Single newline between items
                        content_line = prefix + value
                        f.write(content_line.strip())
                        prev_item_written = True
                        content_written = True


def generate_parquet_links(url_base, start_num, stop_num, total_shards, padding_digits=5):
    """
    Generate a list of parquet file links based on a numerical range.
    """
    links = []
    # Format the total number of shards with padding
    total_str = str(total_shards).zfill(padding_digits)
    
    # Loop from start to stop number (inclusive)
    for i in range(start_num, stop_num + 1):
        # Format the current shard index number with padding
        index_str = str(i).zfill(padding_digits)
        
        # Construct the full URL based on the common pattern
        link = f"{url_base}-{index_str}-of-{total_str}.parquet?download=true"
        links.append(link)
    
    print(f"Generated {len(links)} links, from index {start_num} to {stop_num}.")
    return links


def main(
    url_base,
    start_num,
    stop_num,
    total_shards,
    padding_digits,
    output_text_file,
    include_keys,
    value_prefixes,
    required_key,
    skip_empty,
    exclude,
    append,
    list_key,
    role_prefixes,
):
    # Generate the list of links instead of scraping
    parquet_links = generate_parquet_links(
        url_base, start_num, stop_num, total_shards, padding_digits
    )

    download_dir = "./downloaded_parquets"
    json_dir = "./json_output"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    if not append:
        # Clear the file if not in append mode
        open(output_text_file, "w").close()

    for link in parquet_links:
        file_name = link.split("/")[-1].split("?")[0]  # Extract filename
        parquet_path = os.path.join(download_dir, file_name)
        json_path = os.path.join(json_dir, file_name.replace(".parquet", ".json"))

        # Download the Parquet file if it doesn't already exist
        if not os.path.exists(parquet_path):
            download_file(link, parquet_path)

        # Convert the Parquet file to JSON
        convert_to_json(parquet_path, json_path)

        # Emit the JSON contents and write output to a text file
        emit_json_contents(
            json_path,
            output_text_file,
            include_keys,
            value_prefixes,
            required_key,
            skip_empty,
            exclude,
            list_key=list_key,
            role_prefixes=role_prefixes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and convert a range of Parquet files to JSON and save contents to a text file."
    )

    # --- New arguments for URL generation ---
    parser.add_argument(
        "--url_base",
        type=str,
        required=True,
        help="Base URL for the parquet files, up to (but not including) the shard number. "
             "Example: 'https://huggingface.co/datasets/skymizer/fineweb-edu-dedup-45B/resolve/main/data/train'",
    )
    parser.add_argument(
        "--start_num",
        type=int,
        required=True,
        help="The starting shard number (e.g., 0).",
    )
    parser.add_argument(
        "--stop_num",
        type=int,
        required=True,
        help="The ending shard number (inclusive) (e.g., 468).",
    )
    parser.add_argument(
        "--total_shards",
        type=int,
        required=True,
        help="The total number of shards for the '...-of-YYYYY.parquet' part (e.g., 469).",
    )
    parser.add_argument(
        "--padding_digits",
        type=int,
        default=5,
        help="Number of digits to pad shard numbers with (default: 5, e.g., '00001').",
    )
    
    # --- Existing arguments ---
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file where the contents should be saved.",
    )
    parser.add_argument(
        "-i",
        "--include_keys",
        type=str,
        nargs="+",
        required=True,
        help="List of keys to include from the JSON contents.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        action="append",
        metavar=("KEY", "VALUE"),
        help="Specify key-value pairs to be excluded. Use the format: --exclude KEY VALUE [KEY VALUE ...]",
    )
    parser.add_argument(
        "-p",
        "--value_prefix",
        type=str,
        nargs="+",
        required=True,
        help="List of prefixes to be added to each individual value when emitting to the text file.",
    )
    parser.add_argument(
        "-r",
        "--required_key",
        type=str,
        default=None,
        help="Only emit items that have this key (optional).",
    )
    parser.add_argument(
        "-s",
        "--skip_empty",
        default=False,
        action="store_true",
        help="Skip any item which is the empty string",
    )
    parser.add_argument(
        "-a",
        "--append",
        default=False,
        action="store_true",
        help="Append to the current input.txt file",
    )
    parser.add_argument(
        "--list_key",
        type=str,
        default=None,
        help="If provided, specifies a key whose value is a list to process",
    )
    parser.add_argument(
        "--role_prefixes",
        type=str,
        nargs="+",
        metavar=("ROLE", "PREFIX"),
        help="Specify prefixes for roles. Use the format: --role_prefixes ROLE PREFIX [ROLE PREFIX ...]",
    )
    
    args = parser.parse_args()

    # Pass all arguments to main
    main(
        args.url_base,
        args.start_num,
        args.stop_num,
        args.total_shards,
        args.padding_digits,
        args.output_text_file,
        args.include_keys,
        args.value_prefix,
        args.required_key,
        args.skip_empty,
        args.exclude,
        args.append,
        args.list_key,
        args.role_prefixes,
    )
