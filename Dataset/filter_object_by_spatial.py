import json
import argparse
from typing import List, Dict, Any, Optional
import os

def parse_args(arg_list: Optional[List[str]] = None):
    """
    Command Line Arguments

    --data_path: Data path of sceneXXXX_XX data
    --spatial_relation: Type of spatial relation, options: Above, Below, Closest, Farthest, Between, Near, In, and On
    --object_id: ID of object
    """
    parser = argparse.ArgumentParser(description="Your program description here")

    parser.add_argument(
        "-d", "--data_path",
        type=str,
        default="",
        help="sceneXXXX_XX data path"
    )

    parser.add_argument(
        "-r", "--spatial_relation",
        type=str,
        default="",
        help="Spatial Relation Type"
    )

    parser.add_argument(
        "-o", "--object_id",
        type=str,
        default=None,
        help="Object ID of interest"
    )
    
    args = parser.parse_args(arg_list)
    return args

def main(arg_list: Optional[List[str]] = None) -> None:

    args = parse_args(arg_list)

    if not os.path.exists(args.data_path):
        print("Folder does not exist.")
        exit()

    try:
        # Remove trailing slash if exist
        if args.data_path.endswith("/"):
            args.data_path = args.data_path[:-1]
        
        args.spatial_relation = args.spatial_relation.lower()

        with open(f"{args.data_path}/{args.data_path.split('/')[-1]}_grouped_by_anchor.json", 'r') as f:
            data = json.load(f)["0"]

        with open(f"{args.data_path}/{args.data_path.split('/')[-1]}_objects.json", 'r') as f:
            mapping = json.load(f)["0"]

        if data.get(args.spatial_relation) == None:
            print("Invalid Spatial Relation, please pick from Above, Below, Closest, Farthest, Between, Near, In, and On")
            exit()

        object_name = mapping[args.object_id]["nyu40_label"]

        data = data[args.spatial_relation]

        if data.get(object_name) == None:
            print("[]")
            # return "[]"
        
        data = data[object_name]
        
        if data.get(args.object_id) == None:
            print("[]")
            # return "[]"
        else:
            ids = [v[0] for _, v in data[args.object_id].items()]
            print(ids)
            # return ids

    except Exception as e:
        print("Error: ", e)
        exit()

    


if __name__ == "__main__":
    main()