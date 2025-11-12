from utils import get_object_colors_and_coordinates, create_ambiguous_question, get_env_info_and_stats, get_similar_color, create_nonexistent_objects, get_scanrefer_environment_info, convert_numpy
import os
import argparse
from typing import List, Optional, Any
import json
import numpy as np
import random
from tqdm import tqdm

FILTER_OBJECTS = ['wall', 'floor', 'ceiling', 'object']

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
        nargs='+',
        default=None,
        help="ScanRefer data path(s). Can specify multiple JSON files (e.g., file1.json file2.json)"
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
        default="scannet_data/scans",
        help="ScanNet data path"
    )

    parser.add_argument(
        "-scene_ids", "--scene_ids",
        type=str,
        nargs='+',
        default=None,
        help="Specific scene IDs to process (e.g., scene0000_00 scene0001_00). If not specified, all scenes will be processed."
    )

    parser.add_argument(
        "-scene_range", "--scene_range",
        type=int,
        nargs=2,
        default=None,
        metavar=('START', 'END'),
        help="Process scenes by index range [start, end). E.g., --scene_range 0 100"
    )

    parser.add_argument(
        "-process_id", "--process_id",
        type=str,
        default=None,
        help="Process ID to append to output filename for parallel processing (e.g., 'p0', 'p1')"
    )

    
    args = parser.parse_args(arg_list)
    return args


def main(arg_list: Optional[List[str]] = None) -> None:
    args = parse_args(arg_list)

    if args.mode == "":
        print("Mode not specified")
        exit()
    elif args.mode == "cc":
        if os.path.isdir(args.scannet_data_path):
            files = os.listdir(args.scannet_data_path)
            # Filter out scenes that don't end with _00
            files = [f for f in files if f.endswith('_00')]
            # Filter by scene_ids if specified
            if args.scene_ids:
                files = [f for f in files if f in args.scene_ids]
                print(f"Processing {len(files)} specified scenes: {files}")
            for f in files:
                if not(os.path.isfile(f"{args.scannet_data_path}/{f}/{f}_vh_clean_2.ply") and os.path.isfile(f"{args.scannet_data_path}/{f}/{f}.aggregation.json") and os.path.isfile(f"{args.scannet_data_path}/{f}/{f}_vh_clean_2.0.010000.segs.json")):
                    print(f"Required files for {f} not present, please download the required ScanNet data with 'sh download_scannet_data.sh' first.")
                    continue
                elif os.path.isfile(f"{args.scannet_data_path}/{f}/{f}_cc.aggregation.json"):
                    print(f"Skipping {f}")
                    continue
                colors = get_object_colors_and_coordinates(f"{args.scannet_data_path}/{f}/{f}_vh_clean_2.ply", f"{args.scannet_data_path}/{f}/{f}.aggregation.json", f"{args.scannet_data_path}/{f}/{f}_vh_clean_2.0.010000.segs.json")
                print(f"Loaded files for {f}...")
                scene_objects = json.loads(open(f"{args.scannet_data_path}/{f}/{f}.aggregation.json").read())
                scene_objects['segGroups'] = [{
                    "id": obj["id"],
                    "objectId": obj["objectId"],
                    "segments": obj["segments"],
                    "label": obj["label"],
                    "color": colors[obj["id"]]["color"],
                    "centroid": [float(x) for x in colors[obj["id"]]["centroid"]],
                    "dimensions": [float(x) for x in colors[obj["id"]]["dimensions"]],
                } for obj in scene_objects['segGroups'] if colors.get(obj["id"]) != None]
                with open(f"{args.scannet_data_path}/{f}/{f}_cc.aggregation.json", "w") as f:
                    json.dump(scene_objects, f, indent=4)

        else:
            print(f"The folder '{args.scannet_data_path}' does not exist, please download the required ScanNet data with 'sh download_scannet_data.sh' first.")

    elif args.mode == "ambiguity":
        if args.scanrefer_data_path:
            # Load and merge multiple ScanRefer JSON files
            scanrefer_data = []
            print(f"Loading {len(args.scanrefer_data_path)} ScanRefer data file(s)...")
            for file_path in args.scanrefer_data_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            scanrefer_data.extend(data)
                            print(f"  ✓ Loaded {file_path}: {len(data)} entries")
                        else:
                            print(f"  ⚠ {file_path}: Not a list, skipping")
                except Exception as e:
                    raise ValueError(f"Error loading ScanRefer data from {file_path}: {e}")
            print(f"Total ScanRefer entries loaded: {len(scanrefer_data)}")

        if os.path.isdir(args.scannet_data_path):
            files = os.listdir(args.scannet_data_path)
            # Filter out scenes that don't end with _00
            files = [f for f in files if f.endswith('_00')]
            files.sort()
            
            # Filter by scene_ids if specified
            if args.scene_ids:
                files = [f for f in files if f in args.scene_ids]
                print(f"Processing {len(files)} specified scenes: {files}")
            # Filter by scene_range if specified
            elif args.scene_range:
                start, end = args.scene_range
                files = files[start:min(end, len(files))]
                print(f"Processing scenes [{start}:{end}] ({len(files)} scenes): {files[:5]}{'...' if len(files) > 5 else ''}")
            combined_data = []
            for f in tqdm(files):
                try:
                    if os.path.isfile(f"{args.scannet_data_path}/{f}/{f}_cc.aggregation.json"):
                        data = json.loads(open(f"{args.scannet_data_path}/{f}/{f}_cc.aggregation.json").read())
                        # Environment
                        env_info, object_stats = get_env_info_and_stats(data['segGroups'])

                        # Non-Ambiguity
                        non_multi_objects = [object_name for object_name, object in object_stats.items() if len(object['id']) == 1]
                        non_multi_objects = random.sample(non_multi_objects, min(args.samples_per_scene, len(non_multi_objects)))
                        for object_name in non_multi_objects:
                            if object_name in FILTER_OBJECTS:
                                continue
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
                            if object_name in FILTER_OBJECTS:
                                continue
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

                        # Missing Object Ambiguity
                        objects = random.sample([object for object in object_stats], min(len(object_stats), args.samples_per_scene))

                        for object_name in objects:
                            if object_name in FILTER_OBJECTS:
                                continue
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

                        # Missing Attribute Ambiguity
                        objects = random.sample([object_name for object_name in object_stats], min(args.samples_per_scene, len(object_stats)))

                        for object_name in objects:
                            if object_name in FILTER_OBJECTS:
                                continue
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
                except Exception as e:
                    print(f"Skipping processing scene {f}: {e}")
                    continue
               
            # Generate output filename
            output_path = args.output
            if args.process_id:
                # Insert process_id before file extension
                base, ext = os.path.splitext(args.output)
                output_path = f"{base}_{args.process_id}{ext}"
            
            print(f"Writing to output file {output_path}...")
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2, default=convert_numpy)

        else:
            print(f"The folder '{args.scannet_data_path}' does not exist, please download the required ScanNet data with 'sh download_scannet_data.sh' first.")


if __name__ == "__main__":
    main()
