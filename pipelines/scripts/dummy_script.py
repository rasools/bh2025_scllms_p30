#!/usr/bin/env python3
"""
Simple Python script that processes data based on CLI argument and environment variable.
Produces an output file.
"""
import os
import sys

def main():
    # Get command-line argument
    if len(sys.argv) < 2:
        print("Usage: process_data.py <input_value>")
        sys.exit(1)
    
    input_value = sys.argv[1]
    
    # Get environment variable
    env_var = os.environ.get('PROCESSING_MODE', 'default')
    
    # Process the data
    output_content = f"Input value: {input_value}\n"
    output_content += f"Processing mode (from env): {env_var}\n"
    output_content += f"Processed result: {input_value.upper()}\n"
    
    # Write output file
    output_file = f"output_{input_value}.txt"
    with open(output_file, 'w') as f:
        f.write(output_content)
    
    print(f"Output file created: {output_file}")

if __name__ == "__main__":
    main()

