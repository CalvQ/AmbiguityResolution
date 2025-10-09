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
