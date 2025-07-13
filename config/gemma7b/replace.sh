#!/bin/bash

# Function to search and replace within a file
replace_in_file() {
  local file="$1"
  local old_text="$2"
  local new_text="$3"
  
  # Use sed to perform in-place search and replace
  sed -i "s/$old_text/$new_text/g" "$file"
}

# Find all YAML files recursively
find . -type f -name "*.yaml" -exec  replace_in_file {} "gemma-2b" "gemma-7b" \;

echo "Replacement completed!"

