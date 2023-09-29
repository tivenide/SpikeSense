import sys

def process_data(path_to_mearec_h5, output_path):
    from import_functions import MEArecRecordingH5Importer
    mearec_importer = MEArecRecordingH5Importer(path_to_mearec_h5)
    from application import application
    application(mearec_importer, output_path)

if len(sys.argv) != 3:
    print("Usage: python quick_detection.py <input_path> <output_path>")
else:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process_data(input_path, output_path)