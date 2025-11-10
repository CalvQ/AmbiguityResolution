import json
from pathlib import Path
from arengine import AREngine

def test_single_scene(json_path, scene_index=0):
    """
    Load a single scene from the evaluation dataset and test it interactively
    using resolve_prompt.
    
    Args:
        json_path (str): Path to the JSON dataset
        scene_index (int): Index of the scene to test (default: 0)
    """
    # Load the dataset
    data = json.loads(Path(json_path).read_text())
    
    if scene_index >= len(data):
        print(f"Error: Scene index {scene_index} out of range (dataset has {len(data)} scenes)")
        return
    
    # Get the specific scene
    item = data[scene_index]
    
    # Extract scene information
    scene = {"environment_info": (item.get("environment_info") or "").strip()}
    initial_prompt = item.get("dialogue", [])[0]["text"]
    
    # Display scene information
    print("=" * 70)
    print("SCENE INFORMATION")
    print("=" * 70)
    print(f"Scene ID: {item.get('scene_id')}")
    print(f"Object ID: {item.get('object_id')}")
    print(f"Object Name: {item.get('object_name')}")
    print(f"\nScene Description:\n{scene['environment_info']}")
    print(f"\nInitial Prompt: '{initial_prompt}'")
    print("=" * 70)
    print()
    
    # Initialize the AR engine
    print("Loading model...")
    engine = AREngine()
    engine.load_model()
    engine.set_scene(scene)
    print("Model loaded!\n")
    
    # Run the interactive resolution process
    print("Starting interactive resolution...")
    print("-" * 70)
    
    try:
        final_prompt, history = engine.resolve_prompt(initial_prompt)
        
        # Display results
        print("\n")
        print("=" * 70)
        print("RESOLUTION COMPLETE")
        print("=" * 70)
        print(f"\nInitial Prompt: '{initial_prompt}'")
        
        if history:
            print(f"\nConversation History ({len(history)} turns):")
            for i, msg in enumerate(history, 1):
                print(f"  Turn {i}: {msg}")
        else:
            print("\nNo clarification needed - prompt was unambiguous!")
        
        print(f"\nFinal Resolved Prompt: '{final_prompt}'")
        print(f"\nGround Truth Object: {item.get('object_name')} (ID: {item.get('object_id')})")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during resolution: {e}")


def list_scenes(json_path, max_display=10):
    """
    List available scenes in the dataset.
    
    Args:
        json_path (str): Path to the JSON dataset
        max_display (int): Maximum number of scenes to display
    """
    data = json.loads(Path(json_path).read_text())
    
    print(f"Dataset contains {len(data)} scenes\n")
    print("Available scenes:")
    print("-" * 70)
    
    for i, item in enumerate(data[:max_display]):
        prompt = item.get("dialogue", [])[0]["text"]
        obj_name = item.get("object_name", "unknown")
        scene_id = item.get("scene_id", "unknown")
        
        print(f"[{i}] Scene {scene_id} | Object: {obj_name}")
        print(f"    Prompt: '{prompt}'")
        print()
    
    if len(data) > max_display:
        print(f"... and {len(data) - max_display} more scenes")
    print("-" * 70)


if __name__ == "__main__":
    import sys
    
    # Default dataset path
    default_dataset = "ambiguity_dataset_demo_10.json"
    
    # Parse command line arguments
    # if len(sys.argv) == 1:
    #     # No arguments - show usage
    #     print("Usage:")
    #     print(f"  python test_single_scene.py <dataset.json> <scene_index>")
    #     print(f"  python test_single_scene.py <dataset.json> list")
    #     print()
    #     print("Examples:")
    #     print(f"  python test_single_scene.py {default_dataset} 0")
    #     print(f"  python test_single_scene.py {default_dataset} list")
    #     sys.exit(0)
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else default_dataset
    
    # Check if user wants to list scenes
    if len(sys.argv) > 2 and sys.argv[2].lower() == "list":
        list_scenes(json_path)
    else:
        # Get scene index
        scene_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        test_single_scene(json_path, scene_index)