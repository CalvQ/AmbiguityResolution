# AmbiguityResolution

This repository aims to create a system that can resolve ambiguities from natural language by engaging in multi-turn dialogue with users until ambiguities are resolved.


## Dataset

download ambiguity_data.json in ``data/ambiguity_data.json`` from https://drive.google.com/file/d/1TOGsKY45AjdLhNQKSmb5EsKwvhv-AZzS/view?usp=drive_link.

## Usage of Human Response Agent
Step1: Ensure there are ``ambiguity_data.json`` and ``ScanRefer_filtered.json`` under the data file.

Step2: Before runing ``human-response-agent.py``, replace ``YOUR_OPENAI_API_KEY`` by the valid OpenAI key.

Step3: Run

```
python AREngine/human-response-agent.py \
  --test-sample-limit 50 \
  --round-limit 2 \
  --output-file results/human-response-agent-output.json
```

- ``--test-sample-limit``: Maximum number of samples to test
- ``--round-limit``: Maximum number of dialogue rounds between robot and human
- ``--output-file``: Path to the JSON file where results will be stored