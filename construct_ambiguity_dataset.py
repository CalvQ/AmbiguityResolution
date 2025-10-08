#!/usr/bin/env python3
import json
import os
from collections import defaultdict, Counter
import re
import random
import numpy as np
from typing import Dict, List, Tuple, Set
import functools
import openai

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

client = openai.OpenAI(
    api_key='YOUR_OPENAI_API_KEY',
    base_url="https://ai-gateway.andrew.cmu.edu/"
)

try:
    from plyfile import PlyData
except ImportError:
    print("Please install plyfile: pip install plyfile")
    PlyData = None

FILTER_OBJECTS = ['wall', 'floor', 'ceiling', 'object']
COLOR_OBJECTS = ['red', 'green', 'blue', 'yellow', 'white', 'black', 'orange', 'purple', 'brown', 'silver','gold', 'gray']

COLOR_SIMILARITY_GROUPS = {
    'red': ['orange', 'purple'],   
    'green': ['blue', 'yellow'],      
    'blue': ['green', 'purple'],      
    'yellow': ['green', 'orange'],    
    'white': ['gray', 'silver'],             
    'black': ['purple'],             
    'orange': ['red', 'yellow'],     
    'purple': ['red', 'blue', 'black'],
    'brown': ['orange', 'black'],
    'silver': ['black', 'white', 'gray'],
    'gold': ['yellow', 'orange'],
    'gray': ['black', 'white', 'silver']
}

_scene_env_cache = {}

_gpt_logs = []

def save_gpt_logs(filename: str = "gpt_interaction_logs.json"):
    global _gpt_logs
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(_gpt_logs, f, ensure_ascii=False, indent=2)
    return filename


def cached_build_env_info(scene_id: str, scannet_path: str, scanrefer_data: List[Dict] = None) -> str:

    cache_key = f"{scene_id}"
    
    if cache_key in _scene_env_cache:
        print(f"Using cached environment info for scene {scene_id}")
        return _scene_env_cache[cache_key]
    
    print(f"Building and caching environment info for scene {scene_id}")
    env_info = build_env_info(scene_id, scannet_path, scanrefer_data)
    _scene_env_cache[cache_key] = env_info
    
    return env_info

def clear_env_cache():
    global _scene_env_cache
    _scene_env_cache.clear()
    print("Environment cache cleared")

def get_scene_object_color_combinations(scene_id: str, scanrefer_data: list) -> set:
    scene_color_object_combinations = set()
    
    for item in scanrefer_data:
        if item['scene_id'] == scene_id:
            description = item['description'].lower()
            object_name = item['object_name'].lower()
            
            for color in COLOR_OBJECTS:
                if color in description:
                    color_object_combo = f"{color}_{object_name}"
                    scene_color_object_combinations.add(color_object_combo)
    
    return scene_color_object_combinations

def get_similar_color_safe(ground_truth_color: str, object_name: str, scene_color_object_combinations: set) -> str:
    if ground_truth_color in COLOR_SIMILARITY_GROUPS:
        similar_colors = COLOR_SIMILARITY_GROUPS[ground_truth_color]
        available_similar = []
        for color in similar_colors:
            color_object_combo = f"{color}_{object_name.lower()}"
            if color_object_combo not in scene_color_object_combinations:
                available_similar.append(color)
        
        if available_similar:
            return random.choice(available_similar)
    
    available_colors = []
    for color in COLOR_OBJECTS:
        if color != ground_truth_color:
            color_object_combo = f"{color}_{object_name.lower()}"
            if color_object_combo not in scene_color_object_combinations:
                available_colors.append(color)
    
    if available_colors:
        return random.choice(available_colors)
    else:
        if ground_truth_color in COLOR_SIMILARITY_GROUPS:
            return COLOR_SIMILARITY_GROUPS[ground_truth_color][0]
        else:
            return 'red' if ground_truth_color != 'red' else 'blue'

def get_similar_color(ground_truth_color: str) -> str:
    if ground_truth_color in COLOR_SIMILARITY_GROUPS:
        similar_colors = COLOR_SIMILARITY_GROUPS[ground_truth_color]
        return random.choice(similar_colors)
    else:
        available_colors = [color for color in COLOR_OBJECTS if color != ground_truth_color]
        return random.choice(available_colors)

def read_mesh_vertices_rgb(filename):
    if PlyData is None:
        return None
    
    try:
        with open(filename, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']
            vertices[:,3] = plydata['vertex'].data['red']
            vertices[:,4] = plydata['vertex'].data['green']
            vertices[:,5] = plydata['vertex'].data['blue']
        return vertices
    except Exception as e:
        print(f"Error reading PLY file {filename}: {e}")
        return None

def read_segments_data(scene_id: str, scannet_path: str):
    segs_path = os.path.join(scannet_path, 'scans', scene_id, f'{scene_id}_vh_clean_2.0.010000.segs.json')
    if not os.path.exists(segs_path):
        return None
    
    try:
        with open(segs_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading segments file {segs_path}: {e}")
        return None

def calculate_object_properties(vertices, segments, object_segments):
    if vertices is None or segments is None:
        return None
    
    object_segments_set = set(object_segments)
    
    segments_array = np.array(segments)
    mask = np.isin(segments_array, list(object_segments_set))
    object_vertex_indices = np.where(mask)[0]
    
    if len(object_vertex_indices) == 0:
        return None

    object_vertices = vertices[object_vertex_indices]
    
    center = np.mean(object_vertices[:, :3], axis=0)
    
    min_coords = np.min(object_vertices[:, :3], axis=0)
    max_coords = np.max(object_vertices[:, :3], axis=0)
    size = max_coords - min_coords
    
    avg_color = np.mean(object_vertices[:, 3:6], axis=0)
    
    return {
        'center': center.tolist(),
        'size': size.tolist(),
        'color': avg_color.tolist(),
        'vertex_count': len(object_vertex_indices)
    }

def build_env_info(scene_id: str, scannet_path: str, scanrefer_data: List[Dict] = None) -> str:
    aggregation_data = load_scannet_aggregation(scene_id, scannet_path)
    if not aggregation_data:
        return f"No aggregation data found for scene {scene_id}"
    
    print(f"Loading PLY file for scene {scene_id}...")
    ply_path = os.path.join(scannet_path, 'scans', scene_id, f'{scene_id}_vh_clean_2.ply')
    vertices = read_mesh_vertices_rgb(ply_path)
    if vertices is None:
        return f"Failed to load PLY file for scene {scene_id}"
    
    print(f"Loading segments data for scene {scene_id}...")
    segments_data = read_segments_data(scene_id, scannet_path)
    if not segments_data or 'segIndices' not in segments_data:
        return f"No segments data found for scene {scene_id}"
    
    segments = segments_data['segIndices']
    print(f"Found {len(segments)} segments, processing all objects...")
    
    all_objects = []
    matching_objects = []
    
    for seg_group in aggregation_data['segGroups']:
        label = seg_group.get('label', '').lower()
        if label and label not in FILTER_OBJECTS: 
            object_id = seg_group.get('objectId')
            object_segments = seg_group.get('segments', [])
            
            print(f"Calculating properties for {label} object {object_id} with {len(object_segments)} segments...")
            properties = calculate_object_properties(vertices, segments, object_segments)
            if properties:
                obj_info = {
                    'object_id': object_id,
                    'label': label,
                    'properties': properties
                }
                all_objects.append(obj_info)

                
                print(f"Found {label} object {object_id} with {properties['vertex_count']} vertices")
    
    if not all_objects:
        return f"No objects found in scene {scene_id}"
  
    random.shuffle(all_objects)
    
    # env_info = f"Scene {scene_id} contains {len(all_objects)} objects total, with {len(matching_objects)} {object_label}(s):\n\n"
    env_info = ""
    for i, obj in enumerate(all_objects):
        props = obj['properties']
        center = props['center']
        size = props['size']
        label = obj['label']
        object_id = obj['object_id']
        
        # TODO: more accurate color description
        color_desc = None
        if scanrefer_data:
            color_desc = get_object_color_from_scanrefer(object_id, scene_id, scanrefer_data)
        
        if color_desc is None:
            color = props['color']
            color_desc = get_color_description(color)
        
        size_desc = f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"

        env_info += f"{label.capitalize()}: "
        env_info += f"Center at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
        env_info += f"Size {size_desc}, Color {color_desc}\n"
    
    print(f"Completed processing scene {scene_id}")
    return env_info

def get_scanrefer_environment_info(scene_id: str, scanrefer_data: List[Dict]) -> str:
    scene_objects = [obj for obj in scanrefer_data if obj['scene_id'] == scene_id]
    
    if not scene_objects:
        return f"Scene {scene_id} not found in ScanRefer data."
    
    descriptions = []
    
    for obj in scene_objects:
        object_name = obj['object_name']
        description = obj['description']
        descriptions.append(f"- {object_name}: {description}")
    
    if descriptions:
        env_info = f"Environment description for scene {scene_id}:\n" + "\n".join(descriptions)
    else:
        env_info = f"No object descriptions found for scene {scene_id}."
    
    return env_info

def get_color_description(rgb_color):
    # TODO, maybe call OpenAI API or find a better way to get color description
    r, g, b = rgb_color
    
    if r > 200 and g < 100 and b < 100:
        return "red"
    elif r < 100 and g > 200 and b < 100:
        return "green"
    elif r < 100 and g < 100 and b > 200:
        return "blue"
    elif r > 200 and g > 200 and b < 100:
        return "yellow"
    elif r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > 150 and g > 100 and b < 100:
        return "orange"
    elif r > 100 and g < 100 and b > 100:
        return "purple"
    elif r > 100 and g > 100 and b < 100:
        return "brown"
    else:
        return "mixed color"

def load_scanrefer_data(scanrefer_path: str) -> List[Dict]:
    with open(scanrefer_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_scannet_aggregation(scene_id: str, scannet_path: str) -> Dict:
    aggregation_path = os.path.join(scannet_path, 'scans', scene_id, f'{scene_id}.aggregation.json')
    if not os.path.exists(aggregation_path):
        return None
    
    with open(aggregation_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_object_counts_from_scannet(aggregation_data: Dict) -> Dict[str, int]:
    if not aggregation_data or 'segGroups' not in aggregation_data:
        return {}
    
    object_counts = Counter()
    for seg_group in aggregation_data['segGroups']:
        label = seg_group.get('label', '').lower()
        if label and label not in FILTER_OBJECTS:  
            object_counts[label] += 1
    
    return dict(object_counts)

def simplify_object_name(object_name: str) -> str:
    object_name = object_name.replace('_', ' ')
    if object_name.endswith('s') and len(object_name) > 3:
        return object_name[:-1]
    return object_name

def get_natural_question_for_object(object_name: str, color: str = None) -> str:
    # TODO: not good enough to generate natural questions
    SYSTEM_PROMPT = """
You are generating natural, human-like questions that people would ask when they need to interact with or find an object.
The questions should:
1. Be based on the object's FUNCTION or USE CASE, not just its name
2. Sound natural and conversational, like real human speech
3. Be short and simple (under 15 words)
4. Avoid robotic phrases like "navigate to", "locate", "point to"
5. Express a human need, desire, or situation that requires the object
6. When a color is specified, integrate it naturally into the question (e.g., "I need the red one" or "pass me that blue thing")

Examples WITHOUT color:
- soap dispenser → "I want to wash my hands" / "My hands are dirty, pass me that thing"
- chair → "I'm tired, I need to sit down" / "Guide me to have a seat"
- lamp → "It's too dark here" / "Can you turn on the light?"
- trash can → "Where should I throw this?" / "I need to dispose of this"
- microwave → "I want to heat this up" / "This food is cold"
- bed → "I'm exhausted, I need to lie down" / "Guide me to have a rest"
- refrigerator → "I'm thirsty" / "Where's something cold to drink?"
- window → "I need fresh air" / "Can you open the window?"

Examples WITH color:
- blue chair → "I'm tired, can I sit on the blue one?" / "I need to sit on that blue seat"
- red lamp → "It's dark, turn on the red light" / "Can you switch on the red one?"
- green bottle → "I'm thirsty, pass me the green one" / "Where's that green bottle?"
- white towel → "I need to dry my hands, give me the white one" / "Can I have that white towel?"

Generate ONE natural question for the given object. Only output the question, nothing else.
"""
    
    if color:
        USER_PROMPT = f"Object: {color} {object_name}\nGenerate a natural human question for this colored object."
    else:
        USER_PROMPT = f"Object: {object_name}\nGenerate a natural human question for this object."
    
    answer = get_response(USER_PROMPT, system_prompt=SYSTEM_PROMPT)
    question = answer.strip().strip('"').strip("'")
    return question

def create_ambiguous_question(simple_name: str, 
                                color: str = None, 
                                use_natural: bool = True) -> str:
    if use_natural:
        return get_natural_question_for_object(simple_name, color)
    else:
        ambiguous_questions = [
                f"find the",
                f"where is the",
                f"locate the",
                f"go to the"
            ]
        temp = random.choice(ambiguous_questions)
        if color:
            return f"{temp} {color} {simple_name}"
        else:
            return f"{temp} {simple_name}"

def create_navigation_question(simple_name: str, 
                                use_natural: bool = True) -> str:
    if use_natural:
        return get_natural_question_for_object(simple_name, None)
    else:
        navigation_questions = [
            f"navigate to the {simple_name}",
            f"go to the {simple_name}",
            f"move to the {simple_name}",
            f"approach the {simple_name}"
        ]
        return random.choice(navigation_questions)

def get_response(prompt, model="gpt-4.1-mini", system_prompt=None):
    response = client.chat.completions.create(
            model= model,
            messages = [
                {"role": "system", "content": system_prompt},
                {
                "role": "user",
                "content": prompt,
                }
            ]
    )
    answer = response.choices[0].message.content
    
    return answer


def create_nonexistent_objects(object_name: str, environment_info:str, scene_id: str = None, object_id: str = None):
    SYSTEM_PROMPT = """
You will be given an object name and environment info. 
Your task is to generate ONE AMBIGUOUS ALTERNATIVE OBJECT, and the reason of ambiguity. 
Follow the rules below:
1. The alternative object must be SEMANTICALLY SIMILAR to the given object 
   (e.g., same category, functional substitute, or common synonym),
   but not the same attributes, and actually not in the same scene.
2. Do NOT generate trivial synonyms that do not create real confusion. For example, if the object is "table", then asking "desk" is invalid, 
   because it does not create genuine ambiguity. 
   Only generate objects that people could realistically confuse with the given object. 
3. Generate ambiguous alternative object that DO NOT EXIST in the environment. 
   DO NOT replace the object with one other object that is ALREAY IN the environment. 

Examples:
Negative Examples (invalid, not ambiguous):
- Object: table → Alternative: "desk"  
  Reason: Desk and table are distinct enough that they do not create confusion in this context. 

Positive Examples (valid ambiguous questions):
- Object: chair → Alternative: "stool"  
  Reason: A stool is a type of seat like a chair, so it can be confused as a chair.  
- Object: sofa → Alternative: "divan"  
  Reason: Divan is a type of sofa, so it can be confused as a sofa.
- Object: sofa → Alternative: "chaise longue"  
  Reason: Chaise longue is a type of sofa, so it can be confused as a sofa.
- Object: mug → Alternative: "cup"  
  Reason: Cup is functionally similar to mug, so the two can be mixed up.  
- Object: table → Alternative: "workbench"
  Reason: Workbench is a type of table, so it can be confused as a table.

Strictly follow the output format:
Alternative: <alternative>
Reason: <reason>
    """
    USER_PROMPT = f"Object: {object_name}, Environment info: {environment_info}"
    answer = get_response(USER_PROMPT, system_prompt=SYSTEM_PROMPT)
    
    alternative = answer.split('Alternative: ')[1].split('Reason: ')[0].strip()
    print('-'*10 + 'Alternative' + '-'*10)
    print(alternative)
    print('-'*10)
    try:
        reason = answer.split('Reason: ')[1].strip()
    except:
        reason = "None"
    print('-'*10 + 'REASON' + '-'*10)
    print(reason)
    print('-'*10)
    return {
        'alternative': alternative,
        'reason': reason
    }
   

def alternate_color_description(color_description: str) -> str:
    # TODO

    return color_description

def get_ground_truth_color(description: str) -> str:
    # TODO
    for color in COLOR_OBJECTS:
        if color in description:
            return color
    return None

def get_object_color_from_scanrefer(object_id: str, scene_id: str, scanrefer_data: List[Dict]) -> str:
    for item in scanrefer_data:
        if (item.get('scene_id') == scene_id and 
            item.get('object_id') == object_id):
            description = item.get('description', '').lower()
            for color in COLOR_OBJECTS:
                if color in description:
                    return color
    return None

def load_existing_dataset(existing_dataset_path: str) -> Set[str]:
    if not os.path.exists(existing_dataset_path):
        print(f"Existing dataset file does not exist: {existing_dataset_path}")
        return set()
    
    try:
        with open(existing_dataset_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        processed_scenes = set()
        for item in existing_data:
            if 'scene_id' in item:
                processed_scenes.add(item['scene_id'])
        
        print(f"Loaded {len(processed_scenes)} processed scenes from existing dataset")
        return processed_scenes
    
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        return set()

def construct_ambiguity_dataset(scanrefer_path: str, scannet_path: str, output_path: str, max_scenes: int = 10, positive_ratio: float = 0.5, existing_dataset_path: str = None):
    clear_env_cache()
    print("Loading ScanRefer data...")
    scanrefer_data = load_scanrefer_data(scanrefer_path)
    
    if existing_dataset_path is None and os.path.exists(output_path):
        existing_dataset_path = output_path
        print(f"Output file {output_path} already exists, will use it to skip processed scenes")
    
    already_processed_scenes = set()
    if existing_dataset_path:
        already_processed_scenes = load_existing_dataset(existing_dataset_path)
        print(f"Will skip {len(already_processed_scenes)} already processed scenes")
    
    if max_scenes is not None:
        print(f"Processing first {max_scenes} scenes from {len(scanrefer_data)} ScanRefer data...")
    else:
        print(f"Processing all {len(scanrefer_data)} ScanRefer data...")
    
    dataset = [] 
    processed_scenes = set()
    positive_count = 0
    negative_count = 0
    sample_counter = 0
    
    for item in scanrefer_data:
        if max_scenes is not None and len(processed_scenes) >= max_scenes:
            break
        scene_id = item['scene_id']
        object_name = item['object_name']
        object_id = item['object_id']
        description = item['description']
        
        if scene_id in already_processed_scenes:
            continue
        
        processed_scenes.add(scene_id)
        if object_name.endswith('s'): # skip plural objects, because we could not distinguish them!!
            continue

        aggregation_data = load_scannet_aggregation(scene_id, scannet_path)
        if not aggregation_data:
            continue
    
        object_counts = extract_object_counts_from_scannet(aggregation_data)
        
        simple_name = simplify_object_name(object_name) # convert to single object name
        count = object_counts.get(simple_name, 0) # object_counts is a dictionary of object name and count
        
        should_generate_positive = (sample_counter % 2 == 0)
        sample_counter += 1
        
        # POSITIVE SAMPLE (NO AMBIGUITY) ----------------------------CLEAR REFERENCE--------------------------------
        # Only generate positive sample if there's exactly ONE object of this type
        if should_generate_positive and count == 1:
            clear_question = create_navigation_question(simple_name, use_natural=True)
            robot_response = f"I found the {simple_name}. {description}"
            
            env_info = cached_build_env_info(scene_id, scannet_path, scanrefer_data)
            
            dialogue = {
                "scene_id": scene_id,
                "object_name": object_name,
                "object_id": object_id,
                "original_description": re.sub(r'\s+', ' ', description).strip(),
                "environment_info": env_info,
                "sample_type": "positive",
                "ambiguity_type": "no_ambiguity",
                "dialogue": [
                    {
                        "speaker": "User",
                        "text": clear_question
                    },
                    {
                        "speaker": "Robot",
                        "text": robot_response
                    }
                ]
            }
            
            dataset.append(dialogue)
            positive_count += 1
            print(f"Generated positive sample {positive_count}: {clear_question}")
        elif should_generate_positive and count > 1:
            print(f"Skipped positive sample generation for {simple_name}: {count} objects found in scene")
        
        # NEGATIVE SAMPLES (AMBIGUITY) ----------------------------VARIOUS AMBIGUITY TYPES--------------------------------
        else:
            # MULTIPLE OBJECTS ----------------------------REFERENTIAL AMBIGUITY--------------------------------
            if count > 1:  # there is ambiguity
                # create ambiguity dialogue
                ambiguous_question = create_ambiguous_question(simple_name, None, use_natural=True)
                robot_response = f"There are multiple {simple_name}s, which {simple_name} are you referring to?"
                
                # clean description, strip, replace two consecutive spaces with one space
                description = re.sub(r'\s+', ' ', description).strip()
                clarification_response = f"{description}"
                
                env_info = cached_build_env_info(scene_id, scannet_path, scanrefer_data)
                
                dialogue = {
                    "scene_id": scene_id,
                    "object_name": object_name,
                    "object_id": object_id,
                    "original_description": description,
                    "environment_info": env_info,
                    "sample_type": "negative",
                    "ambiguity_type": "multiple_objects",
                    "dialogue": [
                        {
                            "speaker": "User",
                            "text": ambiguous_question
                        },
                        {
                            "speaker": "Robot", 
                            "text": robot_response
                        },
                        {
                            "speaker": "User",
                            "text": clarification_response
                        }
                    ]
                }
                
                dataset.append(dialogue)
                negative_count += 1
                print(f"Generated negative sample {negative_count} (multiple objects): {ambiguous_question}")
            
            # NONEXISTENT OBJECTS ----------------------------MISSING OBJECT AMBIGUITY--------------------------------
            environment_info = get_scanrefer_environment_info(scene_id, scanrefer_data)
            nonexistent_object_json = create_nonexistent_objects(simple_name, environment_info, scene_id, object_id)
            alternative_object = nonexistent_object_json['alternative']
            reason = nonexistent_object_json['reason']
            navigation_question = create_ambiguous_question(simple_name, alternative_object, use_natural=True)
            
            env_info = cached_build_env_info(scene_id, scannet_path, scanrefer_data)
            
            dialogue = {
                "scene_id": scene_id,
                "object_name": object_name,
                "object_id": object_id,
                "original_description": re.sub(r'\s+', ' ', description).strip(),
                "environment_info": env_info,
                "sample_type": "negative",
                "ambiguity_type": "nonexistent_object",
                "alternative_object": alternative_object,
                "reason": reason,
                "dialogue": [
                    {
                        "speaker": "User",
                        "text": navigation_question
                    },
                ]
            }
            
            dataset.append(dialogue)
            negative_count += 1
            print(f"Generated negative sample {negative_count} (nonexistent object): {navigation_question}")

            # COLOR AMBIGUITY ---------------------------- COLOR AMBIGUITY--------------------------------
            ground_truth_color = get_ground_truth_color(description)
            if any(color in description for color in COLOR_OBJECTS):
                scene_color_object_combinations = get_scene_object_color_combinations(scene_id, scanrefer_data)
                alternative_color = get_similar_color_safe(ground_truth_color, object_name, scene_color_object_combinations)
                alternative_question = create_ambiguous_question(simple_name, alternative_color, use_natural=True)
                env_info = cached_build_env_info(scene_id, scannet_path, scanrefer_data)
                dialogue = {
                    "scene_id": scene_id,
                    "object_name": object_name,
                    "object_id": object_id,
                    "original_description": description,
                    "ground_truth_color": ground_truth_color,
                    "environment_info": env_info,
                    "sample_type": "negative",
                    "ambiguity_type": "color_ambiguity",
                    "dialogue": [
                        {
                            "speaker": "User",
                            "text": alternative_question
                        }
                    ]
                }
                dataset.append(dialogue)
                negative_count += 1
                print(f"Generated negative sample {negative_count} (color ambiguity): {alternative_question}")
        
        processed_scenes.add(scene_id)

    # save dataset
    print(f"\n=== Dataset Generation Summary ===")
    print(f"Total samples generated: {len(dataset)}")
    print(f"Positive samples (no ambiguity): {positive_count}")
    print(f"Negative samples (with ambiguity): {negative_count}")
    print(f"Positive ratio: {positive_count/(positive_count+negative_count)*100:.1f}%")
    print(f"Negative ratio: {negative_count/(positive_count+negative_count)*100:.1f}%")
    print(f"Cache statistics: {len(_scene_env_cache)} scenes cached")
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)    

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        combined_data = existing_data + dataset

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        print(f"Combined dataset saved to: {output_path} (total samples: {len(combined_data)})")
    except Exception as e:
        print(f"Error merging datasets: {e}")
        print(f"Saving new samples to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


def main():
    scanrefer_path = "/Users/yaqi/cmu/course/cap-planning/scanrefer/ScanRefer_filtered_train.json"
    scannet_path = "/Users/yaqi/cmu/course/cap-planning/scannet"
    output_path = "/Users/yaqi/cmu/course/cap-planning/ambiguity_dataset/natural/ambiguity_dataset_temp.json"

    if not os.path.exists(scanrefer_path):
        print(f"ScanRefer file does not exist: {scanrefer_path}")
        return
    
    if not os.path.exists(scannet_path):
        print(f"ScanNet path does not exist: {scannet_path}")
        return

    construct_ambiguity_dataset(scanrefer_path, 
                                scannet_path,
                                output_path,
                                max_scenes=2,
                                positive_ratio=0.5)

if __name__ == "__main__":
    main()
