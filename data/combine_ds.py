#!/usr/bin/env python
import os
import json
import glob
import argparse
from collections import defaultdict
import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Combine partitioned JSONL files into a single file per dataset")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=".",
        help="Directory containing the part*.jsonl files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save combined files (defaults to input_dir)"
    )
    parser.add_argument(
        "--keep_originals", 
        action="store_true",
        help="Keep original part files after combining"
    )
    parser.add_argument(
        "--target", 
        type=str,
        help="Target dataset name to combine (e.g., 'synth_summarization_ds_gpt4o_dedup')"
    )
    return parser.parse_args()

def find_dataset_files(input_dir, target=None):
    """
    Find dataset files matching the target pattern
    If target is specified, only files matching that dataset name will be returned
    """
    if target:
        # Specific target dataset pattern
        pattern = os.path.join(input_dir, f"{target}.part*.jsonl")
        all_files = glob.glob(pattern)
        
        if not all_files:
            logger.error(f"No files matching {target}.part*.jsonl found in {input_dir}")
            return {}
        
        dataset_groups = {target: all_files}
    else:
        # All datasets pattern
        pattern = os.path.join(input_dir, "*.part*.jsonl")
        all_files = glob.glob(pattern)
        
        dataset_groups = defaultdict(list)
        for file_path in all_files:
            basename = os.path.basename(file_path)
            # Split on ".part" to get the dataset name
            dataset_name = basename.split(".part")[0]
            dataset_groups[dataset_name].append(file_path)
    
    return dataset_groups

def combine_jsonl_files(file_list, output_path):
    """Combine multiple JSONL files into a single JSONL file"""
    total_lines = 0
    line_counts = []
    
    # First pass to count lines for progress bar
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
            line_counts.append(line_count)
            total_lines += line_count
    
    # Second pass to combine files
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Create a progress bar
        pbar = tqdm.tqdm(total=total_lines, desc=f"Combining files for {os.path.basename(output_path)}")
        
        for file_path in sorted(file_list, key=lambda x: int(x.split('.part')[1].split('.')[0])):
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # Verify it's valid JSON
                    try:
                        json.loads(line)
                        outfile.write(line)
                        pbar.update(1)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {file_path}")
        
        pbar.close()
    
    return total_lines

def main():
    args = parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find dataset files
    dataset_groups = find_dataset_files(args.input_dir, args.target)
    
    if not dataset_groups:
        return
    
    logger.info(f"Found {len(dataset_groups)} dataset(s) to combine")
    
    # Process each dataset
    for dataset_name, file_list in dataset_groups.items():
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Found {len(file_list)} part files")
        
        # Sort files by part number to maintain order
        file_list.sort(key=lambda x: int(x.split('.part')[1].split('.')[0]))
        
        # Combine the files
        output_path = os.path.join(args.output_dir, f"{dataset_name}.jsonl")
        total_lines = combine_jsonl_files(file_list, output_path)
        
        logger.info(f"Created combined file: {output_path} with {total_lines} lines")
        
        # Delete original files if not keeping them
        if not args.keep_originals:
            for file_path in file_list:
                os.remove(file_path)
                logger.debug(f"Deleted: {file_path}")

if __name__ == "__main__":
    main()
