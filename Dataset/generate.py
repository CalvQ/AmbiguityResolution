from utils import get_object_colors_and_coordinates, create_ambiguous_question, get_env_info_and_stats, get_similar_color, create_nonexistent_objects, get_scanrefer_environment_info, convert_numpy
import os
import argparse
from typing import List, Optional, Any
import json
import numpy as np
import random
from tqdm import tqdm

# FILTER_OBJECTS = ['wall', 'floor', 'ceiling', 'object']

def parse_args(arg_list: Optional[List[str]] = None):
    """
    Command Line Arguments

    --mode: Get colors of objects (colors), Get ambiguity questions (ambiguity)
    --output: Output file path
    --total_sample_limit: Total sample limit
    --samples_per_scene: Samples per scene
    --scanrefer_data_path: ScanRefer data path, if not None, use it to get natural language environment info to generate missing attribute ambiguity questions
    --use_natural: post-process questions to be more natural as what human would ask
    --scannet_data_path: ScanNet data path
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

    parser.add_argument(
        "-t", "--total_sample_limit",
        type=int,
        default=10,
        help="Total sample limit"
    )

    parser.add_argument(
        "-s", "--samples_per_scene",
        type=int,
        default=5,
        help="Samples per scene"
    )

    parser.add_argument(
        "-scanrefer_data_path", "--scanrefer_data_path",
        type=str,
        default="scanrefer/ScanRefer_filtered.json",
        help="ScanRefer data path"
    )

    parser.add_argument(
        "-use_natural", "--use_natural",
        action="store_true",
        default=False,
        help="Use natural language environment info"
    )

    parser.add_argument(
        "-scannet_data_path", "--scannet_data_path",
        type=str,
        default="scannet_data",
        help="ScanNet data path"
    )
    
    parser.add_argument(
        "-id", "--scene_id",
        type=str,
        nargs='+',
        default=None,
        help="Specific Scene ID to generate data on"
    )
    
    args = parser.parse_args(arg_list)
    return args


def main(arg_list: Optional[List[str]] = None) -> None:
    args = parse_args(arg_list)

    if args.mode == "":
        print("Mode not specified")
        exit()
    elif args.mode == "cc":
        print("'cc' Mode deprecated")
        exit()

    elif args.mode == "ambiguity":
        if args.scanrefer_data_path:
            try:
                with open(args.scanrefer_data_path, 'r', encoding='utf-8') as f:
                    scanrefer_data = json.load(f)
            except Exception as e:
                raise ValueError(f"Error loading ScanRefer data: {e}")

        if os.path.isdir(args.scannet_data_path):
            files = os.listdir(args.scannet_data_path)
            files.sort()
            if args.scene_id != None:
                files = args.scene_id
            combined_data = []
            for f in tqdm(files):

                if os.path.isfile(f"{args.scannet_data_path}/{f}/{f}_object_data.json"):
                    data = json.loads(open(f"{args.scannet_data_path}/{f}/{f}_object_data.json").read())

                    # Environment
                    env_info, object_stats = get_env_info_and_stats(data['0'])

                    # Non-Ambiguity
                    non_multi_objects = [object_name for object_name, object in object_stats.items() if len(object['id']) == 1]
                    non_multi_objects = random.sample(non_multi_objects, min(args.samples_per_scene, len(non_multi_objects)))
                    for object_name in non_multi_objects:
                        # if object_name in FILTER_OBJECTS:
                        #     continue
                        dialogue = {
                            "scene_id": f,
                            "object_name": object_name,
                            "object_id": object_stats[object_name]['id'][0],
                            "original_description": "", # TODO
                                "environment_info": env_info,
                                "sample_type": "positive",
                                "ambiguity_type": "no_ambiguity",
                                "natural": args.use_natural,
                                "dialogue": [
                                    {
                                        "speaker": "User",
                                        "text": create_ambiguous_question(object_name, use_natural=args.use_natural)
                                    },
                                ]
                            }

                        combined_data.append(dialogue)
                        if len(combined_data) >= args.total_sample_limit:
                            break
                    if len(combined_data) >= args.total_sample_limit:
                        break


                    # Referential Ambiguity
                    for object_name, object in object_stats.items():
                        # if object_name in FILTER_OBJECTS:
                        #     continue
                        if len(object['id']) > 1:
                            dialogue = {
                                "scene_id": f,
                                "object_name": object_name,
                                "object_id": np.random.choice(object['id'], 1)[0],
                                "original_description": "", # TODO
                                "environment_info": env_info,
                                "sample_type": "negative",
                                "ambiguity_type": "multiple_objects",
                                "natural": args.use_natural,
                                "dialogue": [
                                    {
                                        "speaker": "User",
                                        "text": create_ambiguous_question(object_name, use_natural=args.use_natural)
                                    },
                                ]
                            }

                            combined_data.append(dialogue)
                            if len(combined_data) >= args.total_sample_limit:
                                break
                    if len(combined_data) >= args.total_sample_limit:
                        break

                    # Missing Attribute Ambiguity
                    objects = random.sample([object for object in object_stats], min(len(object_stats), args.samples_per_scene))

                    for object_name in objects:
                        # if object_name in FILTER_OBJECTS:
                        #     continue
                        index = np.random.choice([i for i in range(len(object_stats[object_name]['id']))], 1)[0]
                        instance = object_stats[object_name]
                        id = instance['id'][index]
                        color = instance['color'][index][0] # TODO: choose the first color if there are multiple colors
                        print('color:')
                        print(color)
                        alternative_color = get_similar_color(color) 
                        
                        dialogue = {
                            "scene_id": f,
                            "object_name": object_name,
                            "object_id": id,
                            "original_description": "",
                            "ground_truth_color": color,
                            "environment_info": env_info,
                            "sample_type": "negative",
                            "ambiguity_type": "color_ambiguity",
                            "alternative_color": alternative_color,
                            "natural": args.use_natural,
                            "dialogue": [
                                {
                                    "speaker": "User",
                                    "text": create_ambiguous_question(f"{alternative_color} {object_name}", use_natural=args.use_natural)
                                }
                            ]
                        }
                        combined_data.append(dialogue)
                        if len(combined_data) >= args.total_sample_limit:
                            break
                    if len(combined_data) >= args.total_sample_limit:
                        break

                    # Missing Object Ambiguity
                    objects = random.sample([object_name for object_name in object_stats], min(args.samples_per_scene, len(object_stats)))

                    env_objects = [obj.lower().replace(' ', '') for obj in object_stats]

                    for object_name in objects:
                        # if object_name in FILTER_OBJECTS:
                        #     continue
                        index = np.random.choice([i for i in range(len(object_stats[object_name]['id']))], 1)[0]
                        instance = object_stats[object_name]
                        id = instance['id'][index]
                        color = instance['color'][index]
                        if args.scanrefer_data_path:
                            natural_language_env_info = get_scanrefer_environment_info(f, scanrefer_data)
                            alternative_object_json = create_nonexistent_objects(object_name, natural_language_env_info)
                        else:
                            alternative_object_json = create_nonexistent_objects(object_name, env_info)
                        alternative_object = alternative_object_json['alternative']

                        # Check for substring matches
                        if any([alternative_object in obj for obj in env_objects]) or any([obj in alternative_object for obj in env_objects]):
                            continue

                        reason = alternative_object_json['reason']
                        
                        dialogue = {
                            "scene_id": f,
                            "object_name": object_name,
                            "object_id": id,
                            "original_description": "",
                            "environment_info": env_info,
                            "sample_type": "negative",
                            "ambiguity_type": "nonexistent_object",
                            "natural": args.use_natural,
                            "alternative_object": alternative_object,
                            "reason": reason,
                            "dialogue": [
                                {
                                    "speaker": "User",
                                    "text": create_ambiguous_question(alternative_object, use_natural=args.use_natural)
                                },
                            ]
                        }
                        combined_data.append(dialogue)
                        if len(combined_data) >= args.total_sample_limit:
                            break
                    if len(combined_data) >= args.total_sample_limit:
                        break

                    print(f"""Statistics:
Scenes: {len(files)}
Referential Ambiguity: {len([f for f in combined_data if f['ambiguity_type']=='multiple_objects'])}
Missing Color Ambiguity: {len([f for f in combined_data if f['ambiguity_type']=='color_ambiguity'])}
Missing Object Ambiguity: {len([f for f in combined_data if f['ambiguity_type']=='nonexistent_object'])}
""")

                else:
                    print(f"Required files for {f} not present, please generate the required color and coordinates data with 'python generate -m cc' first.")
            print(f"Writing to output file {args.output}...")
            with open(args.output, "w") as f:
                        json.dump(combined_data, f, ensure_ascii=False, indent=2, default=convert_numpy)

        else:
            print(f"The folder '{args.scannet_data_path}' does not exist, please download the required ScanNet data with 'sh download_scannet_data.sh' first.")

    


if __name__ == "__main__":
    main()
