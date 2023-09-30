# quick_cli.py
import os
import sys
import argparse

# Determine the path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to sys.path
sys.path.append(project_root)

from import_functions import MEArecRecordingH5Importer
from application.application_quick import application_quick
from development.development_workflow_quick import development_workflow_quick

def process_data(input_path, output_path):
    mearec_importer = MEArecRecordingH5Importer(input_path)
    application_quick(mearec_importer, output_path)

def main():
    parser = argparse.ArgumentParser(description='SpikeSense quick CLI')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for the 'process' command
    process_parser = subparsers.add_parser('process', help='Process data')
    process_parser.add_argument('input_path', help='Path to MEArec recording h5 file')
    process_parser.add_argument('output_path', help='Path to save spiketrain h5 file')

    # Subparser for the 'develop' command
    develop_parser = subparsers.add_parser('develop', help='Start development workflow (quick)')

    args = parser.parse_args()

    if args.command == 'process':
        process_data(args.input_path, args.output_path)
    elif args.command == 'develop':
        development_workflow_quick()
    else:
        print("Invalid command. Use 'process' or 'develop'.")

if __name__ == '__main__':
    main()

