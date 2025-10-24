## Ambiguity Dataset Generation from ScanNet Scenes

### Setup
Note: Please run these commands in the Dataset folder

```shell
# Install dependencies
pip install -r requirements.txt
```

```shell
# Run this to get the relevant ScanNet Files (Including files from IRef-VLA and ScanRefer)
sh download_scannet_data.sh
```

Note: Please also add your OPENAI API Key to the utils.py file

<!-- ### Extracting Color and Coordinates of Objects

```shell
python generate.py -m cc
``` -->

### Generating Ambiguious Questions

```shell
# Note: Output file should be a .json file
# Optional: Add -id argument to specify which scenes to use, e.g. -id scene0000_00 scene0001_00 ...

python generate.py -m ambiguity -o {output_file}
```
