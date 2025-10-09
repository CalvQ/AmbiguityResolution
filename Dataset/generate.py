from utils import get_object_colors_and_coordinates, create_ambiguous_question, get_env_info_and_stats, get_similar_color, create_nonexistent_objects, create_navigation_question
import os
import argparse
from typing import List, Optional, Any, Namespace
import json
import numpy as np
import random
from tqdm import tqdm

scannet_data_path = "scannet_data/scans"

SAMPLES_PER_SCENE = 5

def parse_args(arg_list: Optional[List[str]] = None) -> Namespace:
    """
    Command Line Arguments

    --mode: Get colors of objects (colors), Get ambiguity questions (ambiguity)
    --output: Output file path
    """
    parser = argparse.ArgumentParser(description="Your program description here")
    
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="",
        choices=["", "cc", "ambiguity"],
        help="Mode"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="Output file path"
    )
    
    args = parser.parse_args(arg_list)
    return args


def main(arg_list: Optional[List[str]] = None) -> None:
    """
    
    """
    args = parse_args(arg_list)

    if args.mode == "":
        print("Mode not specified")
        exit()
    elif args.mode == "cc":
        if os.path.isdir(scannet_data_path):
            files = os.listdir(scannet_data_path)
            for f in files:
                if not(os.path.isfile(f"{scannet_data_path}/{f}/{f}_vh_clean_2.ply") and os.path.isfile(f"{scannet_data_path}/{f}/{f}.aggregation.json") and os.path.isfile(f"{scannet_data_path}/{f}/{f}_vh_clean_2.0.010000.segs.json")):
                    print(f"Required files for {f} not present, please download the required ScanNet data with 'sh download_scannet_data.sh' first.")
                    continue
                elif os.path.isfile(f"{scannet_data_path}/{f}/{f}_cc.aggregation.json"):
                    print(f"Skipping {f}")
                    continue
                colors = get_object_colors_and_coordinates(f"{scannet_data_path}/{f}/{f}_vh_clean_2.ply", f"{scannet_data_path}/{f}/{f}.aggregation.json", f"{scannet_data_path}/{f}/{f}_vh_clean_2.0.010000.segs.json")
                scene_objects = json.loads(open(f"{scannet_data_path}/{f}/{f}.aggregation.json").read())
                scene_objects['segGroups'] = [{
                    "id": obj["id"],
                    "objectId": obj["objectId"],
                    "segments": obj["segments"],
                    "label": obj["label"],
                    "color": colors[obj["id"]]["color"],
                    "centroid": colors[obj["id"]]["centroid"],
                    "dimensions": colors[obj["id"]]["dimensions"],
                } for obj in scene_objects['segGroups'] if colors.get(obj["id"]) != None]
                with open(f"{scannet_data_path}/{f}/{f}_cc.aggregation.json") as f:
                    json.dump(scene_objects, f, indent=4)

        else:
            print(f"The folder '{scannet_data_path}' does not exist, please download the required ScanNet data with 'sh download_scannet_data.sh' first.")
    
    elif args.mode == "ambiguity":
        if os.path.isdir(scannet_data_path):
            files = os.listdir(scannet_data_path)
            for f in tqdm(files):
                if os.path.isfile(f"{scannet_data_path}/{f}/{f}_cc.aggregation.json"):
                    data = json.loads(open(f"{scannet_data_path}/{f}/{f}_cc.aggregation.json").read())

                    combined_data = []

                    # Environment
                    env_info, object_stats = get_env_info_and_stats(data['segGroups'])

                    # Referential Ambiguity
                    for object in object_stats:
                        if len(object['id']) > 1:
                            dialogue = {
                                "scene_id": f,
                                "object_name": object,
                                "object_id": np.random.choice(object['id'], 1),
                                "original_description": "", # TODO
                                "environment_info": env_info,
                                "sample_type": "negative",
                                "ambiguity_type": "multiple_objects",
                                "dialogue": [
                                    {
                                        "speaker": "User",
                                        "text": create_ambiguous_question(object, use_natural=False)
                                    },
                                ]
                            }

                            combined_data.append(dialogue)

                    # Missing Object Ambiguity
                    objects = random.sample([object for object in object_stats], SAMPLES_PER_SCENE)

                    for object in objects:
                        index = np.random.choice([i for i in range(len(object['id']))], 1)
                        instance = object_stats[object]
                        id = instance['id'][index]
                        color = instance['color'][index]
                        alternative_color = get_similar_color(color) 
                        
                        dialogue = {
                            "scene_id": f,
                            "object_name": object,
                            "object_id": id,
                            "original_description": "",
                            "ground_truth_color": color,
                            "environment_info": env_info,
                            "sample_type": "negative",
                            "ambiguity_type": "color_ambiguity",
                            "alternative_color": alternative_color,
                            "dialogue": [
                                {
                                    "speaker": "User",
                                    "text": create_ambiguous_question(f"{alternative_color} {object}", use_natural=False)
                                }
                            ]
                        }
                        combined_data.append(dialogue)


                    # Missing Attribute Ambiguity
                    objects = random.sample([object for object in object_stats], SAMPLES_PER_SCENE)

                    for object in objects:
                        index = np.random.choice([i for i in range(len(object['id']))], 1)
                        instance = object_stats[object]
                        id = instance['id'][index]
                        color = instance['color'][index]
                        alternative_object_json = create_nonexistent_objects(object, env_info)
                        alternative_object = alternative_object_json['alternative']
                        reason = alternative_object_json['reason']
                        
                        dialogue = {
                            "scene_id": f,
                            "object_name": object,
                            "object_id": id,
                            "original_description": "",
                            "environment_info": env_info,
                            "sample_type": "negative",
                            "ambiguity_type": "nonexistent_object",
                            "alternative_object": alternative_object,
                            "reason": reason,
                            "dialogue": [
                                {
                                    "speaker": "User",
                                    "text": create_navigation_question(alternative_object, use_natural=False)
                                },
                            ]
                        }
                        combined_data.append(dialogue)

                    print(f"""Statistics:
Scenes: {len(files)}
Referential Ambiguity: {len([f for f in combined_data if f['ambiguity_type']=='multiple_objects'])}
Missing Color Ambiguity: {len([f for f in combined_data if f['ambiguity_type']=='color_ambiguity'])}
Missing Object Ambiguity: {len([f for f in combined_data if f['ambiguity_type']=='nonexistent_object'])}
""")

                    print(f"Writing to output file {args.output}...")
                    with open(args.output, "w") as f:
                        json.dump(combined_data, f, ensure_ascii=False, indent=2)

                else:
                    print(f"Required files for {f} not present, please generate the required color and coordinates data with 'python generate -m cc' first.")
        else:
            print(f"The folder '{scannet_data_path}' does not exist, please download the required ScanNet data with 'sh download_scannet_data.sh' first.")

    


if __name__ == "__main__":
    main()
