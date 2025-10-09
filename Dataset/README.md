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
```python
python generate -m cc
```

### Generating Ambiguious Questions
```python
python generate -m ambiguity -o {output_file}
```