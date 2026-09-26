import argparse
import sys

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Find every unique character in a text file and list them one per line."
    )
    parser.add_init_argument = parser.add_argument(
        "file_path", 
        help="Path to the input text file"
    )
    
    args = parser.parse_args()

    try:
        unique_chars = set()
        
        # Read the file and collect unique characters
        with open(args.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for char in line:
                    unique_chars.add(char)
                    
        # Sort and print one per newline
        for char in sorted(unique_chars):
            # Print literal representation so spaces, tabs, and newlines are visible
            print(repr(char))
            
    except FileNotFoundError:
        print(f"Error: The file '{args.file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

