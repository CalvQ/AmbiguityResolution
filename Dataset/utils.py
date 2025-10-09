import numpy as np
import json
from plyfile import PlyData
import webcolors
import math
import random
from typing import List, Optional, Any, Tuple
import os
import openai

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

client = openai.OpenAI(
    api_key='YOUR_OPENAI_API_KEY',
    base_url="https://ai-gateway.andrew.cmu.edu/"
)

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

COLOR_OBJECTS = ['red', 'green', 'blue', 'yellow', 'white', 'black', 'orange', 'purple', 'brown', 'silver','gold', 'gray']
FILTER_OBJECTS = ['wall', 'floor', 'ceiling', 'object']

def get_dominant_color(colors, n_bins=32):
    """Get the most common color using histogram binning."""
    # Bin colors to reduce noise
    binned = (colors // (256 // n_bins)) * (256 // n_bins)
    
    # Find unique colors and their counts
    unique_colors, counts = np.unique(binned, axis=0, return_counts=True)
    
    # Return most common
    dominant_idx = np.argmax(counts)
    return unique_colors[dominant_idx].astype(int)

# def rgb_to_color_name(rgb):
#     """
#     Convert RGB tuple to closest named color using webcolors.
    
#     Args:
#         rgb: tuple or array of (R, G, B) values
    
#     Returns:
#         Name of closest CSS3 color
#     """
#     rgb_tuple = tuple(int(x) for x in rgb)
    
#     try:
#         # Try exact match first
#         return webcolors.rgb_to_name(rgb_tuple)
#     except ValueError:
#         # Find closest color name
#         min_distance = float('inf')
#         closest_name = None
        
#         for name in webcolors.CSS3_NAMES_TO_HEX:
#             color_rgb = webcolors.name_to_rgb(name)
#             distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_tuple, color_rgb))
            
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_name = name
        
#         return closest_name
    
def rgb_to_color_name(rgb):
    """
    Map RGB values (0-255) to the closest named color.
    
    Args:
        r, g, b: Red, green, blue values (0-255)
    
    Returns:
        str: The closest color name
    """
    # Define RGB values for each named color
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'brown': (165, 42, 42),
        'silver': (192, 192, 192),
        'gold': (255, 215, 0),
        'gray': (128, 128, 128)
    }

    r, g, b = tuple(int(x) for x in rgb)
    
    # Calculate Euclidean distance to each color
    min_distance = float('inf')
    closest_color = None
    
    for color_name, (cr, cg, cb) in color_map.items():
        # Euclidean distance in RGB space
        distance = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

def get_object_colors_and_coordinates(ply_path, aggregation_path, segmentation_path):
    """
    Extract colors for each object in a ScanNet scene.
    
    Args:
        ply_path: Path to _vh_clean_2.ply file
        aggregation_path: Path to .aggregation.json file
        segmentation_path: Path to _vh_clean_2.0.010000.segs.json file
    
    Returns:
        Dictionary mapping object_id to color statistics
    """
    
    # Load PLY file with vertex colors
    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex']
    
    # Extract colors and coordinates
    coords = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    
    # Load segmentation (vertex -> segment mapping)
    with open(segmentation_path, 'r') as f:
        seg_data = json.load(f)
        seg_indices = np.array(seg_data['segIndices'])
    
    # Load aggregation (segment -> object mapping)
    with open(aggregation_path, 'r') as f:
        agg_data = json.load(f)
    
    # Build mapping from segment to object
    segment_to_object = {}
    for obj in agg_data['segGroups']:
        obj_id = obj['objectId']
        obj_label = obj['label']
        for seg_id in obj['segments']:
            segment_to_object[seg_id] = {
                'objectId': obj_id,
                'label': obj_label
            }
    
    # Extract colors for each object
    object_colors = {}
    
    for obj in agg_data['segGroups']:
        obj_id = obj['objectId']
        obj_label = obj['label']
        segments = obj['segments']
        
        # Find all vertices belonging to this object
        mask = np.isin(seg_indices, segments)
        obj_colors = colors[mask]
        obj_coords = coords[mask]

        bbox_min = obj_coords.min(axis=0)
        bbox_max = obj_coords.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        centroid = obj_coords.mean(axis=0)
        
        if len(obj_colors) > 0:
            dominant_rgb = get_dominant_color(obj_colors)
            object_colors[obj_id] = {
                "color": rgb_to_color_name(dominant_rgb),
                "centroid": centroid,
                "dimensions": bbox_size
            }
        else:
            object_colors[obj_id] = {
                "color": "UNK",
                "centroid": centroid,
                "dimensions": bbox_size
            }
    
    return object_colors

def get_env_info_and_stats(data: dict) -> Tuple[str, dict]:
    object_stats = {}
    env_info = ""
    for obj in data:
        center = obj['centroid']
        size = obj['dimensions']
        size_desc = f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"
        env_info += f"{obj['label'].capitalize()}: "
        env_info += f"Center at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
        env_info += f"Size {size_desc}, Color {obj['color']}\n"
        
        if obj in FILTER_OBJECTS or obj['color'] == "UNK":
            continue

        if object_stats.get(obj['label']):
            object_stats[obj['label']]['id'].append(obj['id'])
            object_stats[obj['label']]['color'].append(obj['color'])
            object_stats[obj['label']]['coordinates'].append(obj['centroid'])
        else:
            object_stats[obj['label']] = {
                    'id': [obj['id']],
                    'color': [obj['color']],
                    'coordinates': [obj['centroid']],
                }
    
    return env_info, object_stats

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

def get_natural_question_for_object(object_name: str, color: str = None) -> str:
    SYSTEM_PROMPT = """
You are generating natural, human-like questions that people would ask when they need to interact with or find an object.
The questions should:
1. Be based on the object's FUNCTION and simulate real human needs, NOT A NAIVE QUESTION LIKE "where is ..." or "go to ...".
2. Simulate real conversations between humans and robots. 
3. Simulate a human's need and desire to ask a navigation robot to find an object. The goal is to use your generated question to stimulate the robot's ability to detect the user's intention.
4. Be short and simple (under 15 words)
5. When a color is specified, integrate it naturally into the question.
6. AVOID using ambiguous demonstrative pronouns like "that", "this", "it", "those", etc.
7. AVOID USING the original object name in the question.

Examples WITHOUT color:
- soap dispenser → "I want to wash my hands"
- chair → "I'm tired, I need to sit down"
- lamp → "It's too dark here"
- trash can → "Where should I throw the trash?" / "I need to dispose of the trash"
- microwave → "I want to heat the food up" / "The food is cold"
- bed → "I'm exhausted, I need to lie down" / "Guide me to have a rest"
- refrigerator → "I'm thirsty" / "Where's something cold to drink?"
- window → "I need fresh air"

Examples WITH color:
- blue chair → "I'm tired, can I sit on the blue one?" / "I need to sit on that blue seat"
- red lamp → "It's dark, turn on the red light"
- green bottle → "I'm thirsty, pass me the green one"
- brown guitar → "Let's practice an instrument."

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

# TODO: Check get_similar_color_safe function
def get_similar_color(ground_truth_color: str) -> str:
    if ground_truth_color in COLOR_SIMILARITY_GROUPS:
        similar_colors = COLOR_SIMILARITY_GROUPS[ground_truth_color]
        return random.choice(similar_colors)
    else:
        available_colors = [color for color in COLOR_OBJECTS if color != ground_truth_color]
        return random.choice(available_colors)
    
def create_nonexistent_objects(object_name: str, environment_info:str) -> dict:
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

    try:
        reason = answer.split('Reason: ')[1].strip()
    except:
        reason = "None"

    return {
        'alternative': alternative,
        'reason': reason
    }