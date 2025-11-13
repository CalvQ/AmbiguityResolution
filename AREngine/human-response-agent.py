import openai
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataset.utils import get_scanrefer_environment_info
from AREngine.arengine import AREngine


############################# TO EDIT ##############################
TEST_SAMPLE_LIMIT = 50 # maximum number of samples to test
ROUND_LIMIT = 2 # maximum number of rounds of dialogue between robot and human
output_file = "../results/human-response-agent-output.json"
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
####################################################################


os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
client = openai.OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url="https://ai-gateway.andrew.cmu.edu/"
)

scanrefer_data_path = "../data/ScanRefer_train.json"
with open(scanrefer_data_path, 'r', encoding='utf-8') as f:
    scanrefer_data = json.load(f)

def get_response(messages_list, model="gpt-4.1-mini", system_prompt=None):
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    messages.extend(messages_list)
    # print('='*40, 'MESSAGES', '='*40)
    # print(messages)
    response = client.chat.completions.create(
            model= model,
            messages = messages
    )
    answer = response.choices[0].message.content
    # print('='*40, 'ANSWER', '='*40)
    # print(answer)
    return answer


class HumanAgent:
    def __init__(self, original_query: str, 
                        target_object_id: int, 
                        target_object_name:str, 
                        scene_id: str,
                        gt_ambiguity: str,
                        color_prompt=None):
        self.model = "gpt-4.1-mini"
        target_object = [obj for obj in scanrefer_data if (obj['scene_id'] == scene_id and obj['object_id'] == str(target_object_id))]

        env_info = get_scanrefer_environment_info(scene_id, scanrefer_data)
        description_of_object = target_object[0]['description']

        print('='*40, 'HumanAgent', '='*40)
        print('original_query: ', original_query)
        print('target_object_id: ', target_object_id)
        print('target_object_name: ', target_object_name)
        print('scene_id: ', scene_id)
        print("description_of_object: ", description_of_object)
        print("gt_ambiguity: ", gt_ambiguity)
        print("color_prompt: ", color_prompt)

        gt_ambiguity_prompt = {
            "no_ambiguity": ["clear and without any ambiguity", 
                                "However", 
                                "and provide a more specific request"],
            "color_ambiguity": ["ambiguous in color becase you mistakingly asked the object in a wrong color",
                                "Therefore", 
                                "and provide the robot with the correct color"],
            "multiple_objects": ["ambiguous because you were asking a object but there are multiple objects that match your description", 
                                "Therefore", 
                                "and tell the robot the specific object you are referring to"],
            "nonexistent_object": ["ambiguous because you were asking a object that does not exist in the scene", 
                                    "Therefore", 
                                    "and tell the robot the real object you are referring to"]
        }
        self.system_prompt = f"""
        You are a human in a scene, helping a robot to find your target object. You have already made a query to one robot. Your original request is: {original_query}
        That query was {gt_ambiguity_prompt[gt_ambiguity][0]}.
        {gt_ambiguity_prompt[gt_ambiguity][1]} the robot may be difficult to understand your request, or think your request is ambiguous.
        You need to help the robot understand your request, {gt_ambiguity_prompt[gt_ambiguity][2]}.
        - The TARGET OBJECT you are referring to is: {target_object_name}
        {color_prompt or ""}
        - The DESCRIPTION of your target object is: {description_of_object}
        
        You will be given a response from the robot, and you need to help the robot understand your request, and provide a more specific request, e.g. provide the description of your target object.
        Be concise and helpful.
        """
        self.history = []
        self.original_query = original_query
        self.target_object_id = target_object_id
        self.scene_id = scene_id


    def get_human_response(self, ambiguity_robot_response):
        # add the robot response to the history  (here, ambiguity_robot_response = user, human = assistant)
        self.history.append({
            "role": "user",
            "content": ambiguity_robot_response
        })
        output = get_response(self.history, self.model, self.system_prompt)
        self.history.append({
            "role": "assistant",
            "content": output
        })
        return output

    

if __name__ == '__main__':
    amb_dataset_path = '../data/ambiguity_data.json' 
    with open(amb_dataset_path, 'r', encoding='utf-8') as f:
        amb_dataset = json.load(f)
    
    engine = AREngine()
    engine.load_model()
    
    TEST_COUNT = 0
    
    results = []

    for q in amb_dataset:
        TEST_COUNT += 1
        try:
            color_prompt = None
            if q['ambiguity_type'] == 'color_ambiguity':
                color_prompt = "- The color of target object: " + q['ground_truth_color']
            ha = HumanAgent(original_query = q['dialogue'][0]['text'],
                            target_object_id = q['object_id'],
                            target_object_name = q['object_name'],
                            scene_id = q['scene_id'],
                            gt_ambiguity = q['ambiguity_type'],
                            color_prompt = color_prompt)
        except:
            continue # TODO: WHAT IF NOT IN THE SCANREFER DATASET? USE THE SCANNET INFO.

        
        scene = {"environment_info": q['environment_info']}
        engine.set_scene(scene)
        
        RESOLVED_FLAG = False
        robot_history = []
        current_prompt = q['dialogue'][0]['text']
        dialogue_rounds = []

        count = 0
        while True:
            robot_response = engine.detect_and_ask(current_prompt,robot_history)
            if robot_response != 'Non-Ambiguous':
                count += 1
                robot_history.append({
                    'user': current_prompt,
                    'assistant': robot_response
                })
                human_response = ha.get_human_response(robot_response)
                dialogue_rounds.append({
                    'robot': robot_response,
                    'human': human_response
                })
                current_prompt = human_response
                print('-'*20, 'Round', count, '-'*20)
                print('\n[Robot Clarification]: ', robot_response)
                print('\n[Human Response]: ', human_response)
            else:
                # successfully resolved the ambiguity
                RESOLVED_FLAG = True
                break
            if count == ROUND_LIMIT:
                break

        results.append({
            'scene_id': q['scene_id'],
            'object_id': q['object_id'],
            'object_name': q['object_name'],
            'ambiguity_type': q['ambiguity_type'],
            'initial_query': q['dialogue'][0]['text'],
            'rounds_executed': count,
            'resolved_within_round_limit': RESOLVED_FLAG and count <= ROUND_LIMIT,
            'dialogue_rounds': dialogue_rounds
        })

        if TEST_SAMPLE_LIMIT is not None and TEST_COUNT > TEST_SAMPLE_LIMIT:
            break

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)