## Ambiguity Dataset Generation from ScanNet Scenes

### Setup

```shell
# Install dependencies
pip install -r requirements.txt
```

```shell
# Run this to get the relevant ScanNet Files
sh download_scannet_data.sh
```

Note: Please also add your OPENAI API Key to the utils.py file

### Extracting Color and Coordinates of Objects

```shell
python generate.py -m cc
```

### Generating Ambiguious Questions

```shell
python generate.py -m ambiguity -o {output_file}
```

if want to use scanrefer's description as the input of LLM to generate missing attribute ambiguity questions, download from [scanrefer](https://drive.google.com/file/d/1x9PcZctaLLC79vF42ktl-bNRnixrKO15/view?usp=sharing), unzip files, then run

```shell
python generate.py -m ambiguity -o {output_file} -scanrefer_data_path {your_scanrefer_data_path}
