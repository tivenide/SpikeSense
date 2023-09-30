import os
import sys

# Determine the path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to sys.path
sys.path.append(project_root)


def process_data(path_to_mearec_h5, output_path):
    from import_functions import MEArecRecordingH5Importer
    mearec_importer = MEArecRecordingH5Importer(path_to_mearec_h5)
    from application.application_pretrained_tm import application_pretrained_tm
    application_pretrained_tm(mearec_importer, output_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python spike_detection_cli.py <input_path> <output_path>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        process_data(input_path, output_path)