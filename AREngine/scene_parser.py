import re
import json

def parse_scene_object(line):
    """
    Parse a single line of environment info into a structured object.
    
    Example input:
        "Window: Center at (4.42, 8.26, 1.90), Size 1.66 x 0.64 x 0.57, Color ['dimgray', 'darkslategray', 'darkgray']"
    
    Returns:
        dict: Structured object with type, center, size, and colors
    """
    # Extract object type (everything before the first colon)
    type_match = re.match(r"^([^:]+):", line)
    if not type_match:
        return None
    
    object_type = type_match.group(1).strip()
    
    # Extract center coordinates
    center_match = re.search(r"Center at \(([0-9.]+), ([0-9.]+), ([0-9.]+)\)", line)
    center = None
    if center_match:
        center = {
            "x": float(center_match.group(1)),
            "y": float(center_match.group(2)),
            "z": float(center_match.group(3))
        }
    
    # Extract size dimensions
    size_match = re.search(r"Size ([0-9.]+) x ([0-9.]+) x ([0-9.]+)", line)
    size = None
    if size_match:
        size = {
            "length": float(size_match.group(1)),
            "width": float(size_match.group(2)),
            "height": float(size_match.group(3))
        }
    
    # Extract colors
    color_match = re.search(r"Color \[([^\]]+)\]", line)
    colors = []
    if color_match:
        color_str = color_match.group(1)
        # Parse the color list (handles both 'color' and "color" formats)
        colors = [c.strip().strip("'\"") for c in color_str.split(",")]
    
    return {
        "type": object_type,
        "center": center,
        "size": size,
        "colors": colors
    }


def parse_environment_info(env_info_str, filter_generic=True):
    """
    Parse environment info string into a list of structured objects.
    
    Args:
        env_info_str (str): Raw environment info string from dataset
        filter_generic (bool): If True, remove generic "Object" entries
    
    Returns:
        list[dict]: List of parsed scene objects
    """
    if not env_info_str:
        return []
    
    objects = []
    for line in env_info_str.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        obj = parse_scene_object(line)
        if obj:
            # Filter out generic objects if requested
            if filter_generic and obj["type"].lower() == "object":
                continue
            objects.append(obj)
    
    return objects


def format_object_for_prompt(obj):
    """
    Format a structured object into a concise string for the prompt.
    
    Args:
        obj (dict): Structured object with type, center, size, colors
    
    Returns:
        str: Formatted string representation
    """
    parts = [obj["type"]]
    
    if obj.get("colors"):
        # Use primary color (first in list)
        parts.append(f"({obj['colors'][0]})")
    
    if obj.get("center"):
        center = obj["center"]
        parts.append(f"at ({center['x']:.1f}, {center['y']:.1f}, {center['z']:.1f})")
    
    if obj.get("size"):
        size = obj["size"]
        # Add size description if significantly different dimensions
        dimensions = [size["length"], size["width"], size["height"]]
        max_dim = max(dimensions)
        min_dim = min(dimensions)
        if max_dim > min_dim * 2:  # If one dimension is 2x larger
            if size["height"] > size["length"] and size["height"] > size["width"]:
                parts.append("(tall)")
            elif size["length"] > size["width"] * 1.5 or size["width"] > size["length"] * 1.5:
                parts.append("(long)")
    
    return " ".join(parts)


def scene_to_json_string(scene_objects):
    """
    Convert scene objects list to a formatted JSON string.
    
    Args:
        scene_objects (list[dict]): List of structured scene objects
    
    Returns:
        str: JSON string representation
    """
    return json.dumps(scene_objects, indent=2)


def scene_to_compact_string(scene_objects):
    """
    Convert scene objects to a compact, readable string for prompts.
    
    Args:
        scene_objects (list[dict]): List of structured scene objects
    
    Returns:
        str: Compact string representation
    """
    if not scene_objects:
        return "Empty scene"
    
    lines = []
    for obj in scene_objects:
        lines.append(format_object_for_prompt(obj))
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the parser
    test_env_info = """Window: Center at (4.42, 8.26, 1.90), Size 1.66 x 0.64 x 0.57, Color ['dimgray', 'darkslategray', 'darkgray']
Window: Center at (8.06, 2.13, 1.53), Size 0.64 x 0.31 x 0.67, Color ['lightblue', 'skyblue', 'rosybrown']
Table: Center at (4.49, 6.88, 0.87), Size 1.16 x 2.54 x 1.21, Color ['darkolivegreen', 'saddlebrown', 'sienna']
Object: Center at (6.05, 2.95, 0.80), Size 0.47 x 0.80 x 1.17, Color ['darkslategray', 'black', 'gray']
Sink: Center at (7.77, 3.45, 0.63), Size 0.53 x 0.61 x 0.90, Color ['gray', 'rosybrown', 'white']"""
    
    # Parse with filtering
    objects = parse_environment_info(test_env_info, filter_generic=True)
    
    print("Parsed objects (generic filtered):")
    print(json.dumps(objects, indent=2))
    
    print("\n" + "="*70)
    print("Compact string format:")
    print(scene_to_compact_string(objects))