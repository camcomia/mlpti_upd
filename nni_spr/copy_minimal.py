import os
import shutil
import argparse
import ast

def copy_files_with_structure(input_filepath, output_base_dir):
    """Minimal script to copy files from a list into a structured directory."""
    try:
        with open(input_filepath, 'r') as f:
            content = f.read()
            source_files = ast.literal_eval(content)
            if not isinstance(source_files, list):
                # Keep minimal error checking for list parsing
                raise ValueError("Input file content is not a valid Python list.")
    except FileNotFoundError:
        print(f"Error: Input file not found: '{input_filepath}'")
        return
    except Exception as e:
        print(f"Error reading/parsing input file '{input_filepath}': {e}")
        return

    print(f"Processing {len(source_files)} files -> '{output_base_dir}'")

    for source_path in source_files:
        try:
            parent_dir = os.path.dirname(source_path)
            identifier = os.path.basename(parent_dir)
            filename = os.path.basename(source_path)
            dest_dir = os.path.join(output_base_dir, identifier)
            dest_path = os.path.join(dest_dir, filename)

            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(source_path, dest_path) # Directly copy without checking source exists

        except Exception as e:
            # Minimal error reporting during copy
            print(f"Error processing '{source_path}': {e}")

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files from list into structured dir (minimal).")
    parser.add_argument(
        "input_file",
        help="Path to file containing the list of source paths (as Python list string)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="main",
        help="Base directory for output structure."
    )
    args = parser.parse_args()
    copy_files_with_structure(args.input_file, args.output_dir)