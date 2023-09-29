# SpikeSense: Neurophysiological Data Processing Workflow

This repository contains the codebase for an efficient workflow designed for processing electrophysiological data. The workflow is specifically tailored for analyzing in silico generated data from [MEArec](https://github.com/SpikeInterface/MEArec).

## Functionalities

- **Data Processing:** Our workflow facilitates seamless processing of electrophysiological data originating from [MEArec](https://github.com/SpikeInterface/MEArec). This includes preprocessing steps to prepare data for further analysis.

- **AI-based Spike Detection:** The core of this project features a powerful Transformer model customized for spike detection in neurophysiological time series data. This allows advanced analysis and identification of relevant events in the data.

## Usage

### Installation

To use the workflow, follow these steps:

1. Clone the repository: `git clone https://github.com/tivenide/SpikeSense`
2. Install the required dependencies: `pip install -r requirements.txt`

### Data Processing

Execute data processing with the following command:

```bash
python quick_detection.py path/to/MEArec_recording_input_data.h5 path/to/spike_trains_output_data
