#!/usr/bin/env python
"""
Script to fix multiline f-strings in Python files.

This script finds f-strings with line breaks inside curly braces
and converts them to single-line f-strings that are compatible with
older Python versions.

Usage:
    python fix_fstrings.py [file1 file2 ...]

If no files are specified, it will scan all Python files in the current directory.
"""

from pathlib import Path
import re
import sys

def fix_multiline_fstrings(file_path):
    """Fix multiline f-strings in a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Pattern to find multiline f-strings with line breaks inside {}
        pattern = r'f"([^"]*)\{([^}]*)\n([^}]*)\}([^"]*)"'
        
        # Replace with single-line f-strings
        def replace(match):
            prefix = match.group(1)
            expr_start = match.group(2).strip()
            expr_end = match.group(3).strip()
            suffix = match.group(4)
            return f'f"{prefix}{{{expr_start} {expr_end}}}{suffix}"'
        
        modified_content = re.sub(pattern, replace, content)
        
        # Only write if changes were made
        if content != modified_content:
            with open(file_path, 'w') as f:
                f.write(modified_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Fix specific files
        files = [Path(f) for f in sys.argv[1:]]
    else:
        # Find all Python files in the project
        files = list(Path('.').glob('**/*.py'))
        # Skip virtual environment
        files = [f for f in files if 'nirs/lib' not in str(f)]
    
    fixed_count = 0
    for file in files:
        if fix_multiline_fstrings(file):
            fixed_count += 1
            print(f"Fixed: {file}")
    
    print(f"Fixed {fixed_count} files") 