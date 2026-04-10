import argparse
import os


def divide_file(input_file, output_dir, chunk_size_mb=50):
    if chunk_size_mb <= 0:
        raise ValueError("chunk_size_mb must be greater than 0")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_size = max(1, int(chunk_size_mb * 1024 * 1024))  # Convert MB to bytes
    part_num = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        chunk = f.read(chunk_size)
        while chunk:
            part_num += 1
            output_path = os.path.join(output_dir, f'part_{part_num}.txt')
            with open(output_path, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)
            chunk = f.read(chunk_size)

    print(f"Divided into {part_num} parts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide a large text file into smaller files.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--output_dir", type=str, default="partitioned_file", help="Directory to save the divided files.")
    parser.add_argument(
        "--chunk_size_mb",
        type=float,
        default=50,
        help="Maximum size of each divided file in MB (supports decimals, e.g. 12.5).",
    )

    args = parser.parse_args()
    divide_file(args.input_file, args.output_dir, args.chunk_size_mb)
