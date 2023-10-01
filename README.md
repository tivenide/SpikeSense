# SpikeSense: Neurophysiological Data Processing Workflow

This repository contains the codebase for an efficient workflow designed for processing electrophysiological data. The workflow is specifically tailored for analyzing in silico generated data from [MEArec](https://github.com/SpikeInterface/MEArec).

## Functionalities

- **Data Processing:** The workflow facilitates seamless processing of electrophysiological data originating from [MEArec](https://github.com/SpikeInterface/MEArec). This includes preprocessing steps to prepare data for further analysis.

- **AI-based Spike Detection:** The core of this project features a powerful Transformer model customized for spike detection in neurophysiological time series data. This allows advanced analysis and identification of relevant events in the data.

- **Evaluation and visualization:** Evaluation and visualization functions for further investigation of the development and analyzing process.

- **Import:** Handling different file formats in order to apply the trained model.

## Usage

### Installation

The project was developed under Python 3.9 and Linux Ubuntu 22.04 lts. For machine learning tasks a GPU is recommended.

To use the workflow, follow these steps:

1. Clone the repository: `git clone https://github.com/tivenide/SpikeSense`
2. Install the required dependencies: `pip install -r requirements.txt`

Please install MEArec separately and follow their instructions on: https://mearec.readthedocs.io/en/latest/installation.html

### Data Processing (pretrained Transformer)

Execute data processing with the following command:

```bash
python cli/spike_detection.py path/to/MEArec_recording_input_data.h5 path/to/spike_trains_output_data
```

### Quick start
Only for demonstration purposes to better understand the workflow and check the installation. Please adjust your MEArec data according to your requirement.
#### Data generation with MEArec
1. Use `data_generation_MEArec/MEArec_data_generation_quick.py` to generate some demo files.
2. Put the recordings for training into `quick/MEArec_training_data_recordings`.
#### Model development
Execute development (quick) with the following command:
```bash
python cli/quick_cli.py develop
```
#### Model application
Execute data processing with the following command:
```bash
python  cli/quick_cli.py process path/to/MEArec_recording_input_data.h5 path/to/spike_trains_output_data
```
## Future Work
- choosing license

## License
Copyright 2023 tivenide. All rights reserved.
