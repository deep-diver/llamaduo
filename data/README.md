# Synthetic Dataset from LlamaDuo

1. **synth_classification_ds_gpt4o_dedup**: GPT4o generated 128K synthetic dataset on classification task
2. **synth_closedqa_ds_gpt4o_dedup**: GPT4o generated 128K synthetic dataset on closedqa task
3. **synth_coding_ds_gpt4o_dedup**: GPT4o generated 128K synthetic dataset on coding task
4. **synth_summarize_ds_gpt4o_dedup**: GPT4o generated 256K synthetic dataset on summarize task
5. **synth_summarize_ds_gemini1_5flash_dedup**: Gemini 1.5 Flash generated 256K synthetic dataset on summarize task
6. **synth_summarize_ds_claude3sonnet_dedup**: Claude 3 Sonnet generated 256K synthetic dataset on summarize task

Since the dataset is large, they are splitted into multi-parts. Use be below tool to combine.

# Dataset Combining Tool

This directory contains tools for managing and combining JSONL datasets.

## `combine_ds.py` Script

The `combine_ds.py` script combines multiple partitioned JSONL files into a single file per dataset.

### Usage

```bash
python combine_ds.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--keep_originals] [--target TARGET_NAME]
```

### Arguments

- `--input_dir`: Directory containing the part*.jsonl files (default: current directory)
- `--output_dir`: Directory to save combined files (defaults to input_dir if not specified)
- `--keep_originals`: Flag to keep original part files after combining (by default, originals are deleted)
- `--target`: Target dataset name to combine (e.g., 'synth_summarization_ds_gpt4o_dedup'). If not specified, all datasets in the directory will be combined.

### Examples

Combine all datasets in the current directory:
```bash
python combine_ds.py
```

Combine only a specific dataset:
```bash
python combine_ds.py --target synth_summarization_ds_gpt4o_dedup
```

Combine a specific dataset from a specific directory and save to another:
```bash
python combine_ds.py --input_dir /path/to/input --output_dir /path/to/output --target synth_summarization_ds_gpt4o_dedup
```

Combine datasets but keep the original files:
```bash
python combine_ds.py --keep_originals
```

### How It Works

1. If a target dataset is specified, the script searches for all files matching the pattern `{target}.part*.jsonl` in the input directory.
2. If no target is specified, it searches for all files matching the pattern `*.part*.jsonl` and groups them by dataset name (the part before `.part` in the filename).
3. For each dataset, it combines all part files in numerical order into a single output JSONL file.
4. The combined file will be named `{dataset_name}.jsonl`.
5. Progress is shown with a progress bar for each dataset.

### Requirements

- Python 3.6+
- tqdm (for progress bars)
