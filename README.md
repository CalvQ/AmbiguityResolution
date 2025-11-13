# AmbiguityResolution

This repository aims to create a system that can resolve ambiguities from natural language by engaging in multi-turn dialogue with users until ambiguities are resolved.


## Dataset

download ambiguity_data.json in ``data/ambiguity_data.json`` from https://drive.google.com/file/d/1TOGsKY45AjdLhNQKSmb5EsKwvhv-AZzS/view?usp=drive_link.

## Usage of Human Response Agent
Step1: Ensure there are ``ambiguity_data.json`` and ``ScanRefer_filtered.json`` under the data file.

Step2: Edit the parameters at the begining of the ``AREngine/human-response-agent.py``:

- replace ``YOUR_OPENAI_API_KEY`` in the ``human-response-agent.py``

- ``TEST_SAMPLE_LIMIT``: to control the maximum number of samples to test.

- ``ROUND_LIMIT``: to control the maximum number of dialogue rounds.

- ``output_file``: output file name of evaluation results.

Step3: run 

```
cd AREngine
python human-response-agent.py
```
