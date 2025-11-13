#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from transformers import AutoModelForCausalLM, AutoTokenizer
from AREngine import arengine

class AmbiguityLM:
    def __init__(self):
        rospy.init_node('ambig_lm')
        
        # Subscribers for incoming questions
        self.question_sub = rospy.Subscriber('/question', String, self.question_callback)
        
        # Publishers for answers
        self.answer_pub = rospy.Publisher('/answer', String, queue_size=10)
        
        # Load your model
        self.ambig_res = arengine.AREngine()
        self.ambig_res.load_model()
        
        # Load scene data
        self.ambig_res.set_scene(self.load_scene_data())
    

    def get_env_info_and_stats(data):
        env_info = ""
        for id in data:
            obj = data[id]
            center = obj['center']
            size = obj['size']
            size_desc = f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"
            env_info += f"{obj['raw_label'].capitalize()}: "
            env_info += f"Center at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
            env_info += f"Size {size_desc}, Color {obj['color_labels'][0]}\n"

        return env_info


    def load_scene_data(self):
        """Load preprocessed ScanNet scene data"""
        default_scene_path = "CMU-VLA-Challenge/system/unity/src/vehicle_simulator/mesh/unity/scene0000_00"
        scene_path = rospy.get_param('~scene_path', default_scene_path)

        if scene_path.endswith('/'):
            scene_path = scene_path[:-1]

        scene = scene_path.split('/')[-1]
        
        # Load scene info
        import json
        with open(f"{scene_path}/{scene}_object_data.json", 'r') as f:
            objects = json.load(f)['0']
            return self.get_env_info_and_stats(objects)
    

    def generate_answer(self, question):
        # TODO: Port over model logic here
        self.ambig_res.resolve_prompt(question)

        
    def question_callback(self, msg):
        """Process natural language question"""
        question = msg.data
        answer = self.generate_answer(question)
        self.answer_pub.publish(String(data=answer))


if __name__ == '__main__':
    model = AmbiguityLM()
    rospy.spin()