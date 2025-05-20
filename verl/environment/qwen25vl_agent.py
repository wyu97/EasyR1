import torch
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info, smart_resize
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor
from PIL.Image import Image as ImageObject
import io 
from io import BytesIO
import base64
from typing import Any, Dict, List, Optional, Union
import math
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import torch_functional as VF
from verl.utils.dataset import collate_fn
from verl.protocol import DataProto

SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 532x1148.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Perform a key event on the mobile device.\n    - This supports adb's `keyevent` syntax.\n    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\".\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `system_button`: Press the system button.\n* `open`: Open an app on the device.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, and `action=open`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

SYSTEM_PROMPT_COT = """You are a helpful assistant.

# Google Account
Email: tencent.test.bot@gmail.com
Password: tencenttestbot123!

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1092x2408.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Perform a key event on the mobile device.\n    - This supports adb's `keyevent` syntax.\n    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\".\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `answer`: Output the answer.\n* `system_button`: Press the system button.\n* `open`: Open an app on the device.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate", "answer"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=answer`, and `action=open`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

def process_image_gpt(image):
    #image = Image.open(image_path)
    #print ('before', image.size)
    #image = image.resize((image.width // 2, image.height // 2))
    #print ('after', image.size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return {"url": f"data:image/png;base64,{image_base64}"}

def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    #image = image.resize((image.width // 2, image.height // 2))
    re_w, re_h = smart_resize(image.width, image.height)
    image = image.resize((re_w, re_h))
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

class Qwen25VLAgent(torch.nn.Module):
    def __init__(self, policy_lm = "qwen2_5_vl", max_prompt_len=6144, truncation="error", critic_lm = None, 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = None):
        super(Qwen25VLAgent, self).__init__()
        # if use_bfloat16:
        #     self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir,torch_dtype=torch.bfloat16).to(device)
        # else:
        #     self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir, torch_dtype=torch.float16).to(device)
        # print ('model type', self.model.dtype, self.model.device)

        self.policy_lm = policy_lm
        self.max_prompt_len = max_prompt_len
        self.truncation = truncation
        self.target_critic = None
        self.processor = AutoProcessor.from_pretrained(policy_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def construct_prompt(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT_COT}]
        messages.append({"role": "user", "content": [{'type': 'text', 'text': "The user query: " + observation['task']+'\n\n'}]})
        if len(observation['history']) > 0:
            messages[-1]['content'].append({'type': 'text', 'text': "Task progress (You have done the following operation on the current device):\n"})
            for i, his in enumerate(observation['history']):
                if '<conclusion>' in his and '</conclusion>' in his:
                    conclusion_part = his.split('<conclusion>')[1].split('</conclusion>')[0].strip()
                else:
                    conclusion_part = 'Omitted'
                messages[-1]['content'].append({'type': 'text', 'text': f"Step {i+1}: {conclusion_part};\n"})
        messages[-1]['content'].append({'type': 'text', 'text': "Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags."})
        #print (messages)
        messages[-1]['content'].append({'type': 'image', 'image': process_image(observation['image'], 4194304, 262144)})
        #messages[-1]['content'].append({'type': 'image', 'image': process_image(observation['image'], 4194304, 262144)})
        return messages

    def get_action_inputs(self, observation):
        batch_messages = [self.construct_prompt(ob) for ob in observation]
        #with torch.no_grad():
        batch_text = self.processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        #print ('batch_text.shape', len(batch_text))
        #print (batch_text[0])
        # image_inputs, video_inputs = process_vision_info(batch_messages)
        # inputs = self.processor(
        #     text=text,
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # print (inputs)
        # print (batch_messages[0][-1]['content'][-1]['image'].size, batch_messages[0][-1]['content'][-1]['image'].mode)
        # img = batch_messages[0][-1]['content'][-1]['image'].convert("RGB")
        # print (img.size, img.mode)
        batch = []
        for (msg, text) in zip(batch_messages, batch_text):
            row_dict = {}
            row_dict["multi_modal_data"] = {'image': [msg[-1]['content'][-1]['image']]}
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], text, return_tensors="pt")

            #print (row_dict["multi_modal_data"]['image'][0].height,row_dict["multi_modal_data"]['image'][0].width)
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            #print (input_ids.shape)
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(self.processor, input_ids=input_ids,image_grid_thw=model_inputs.image_grid_thw, attention_mask=attention_mask)  # (3, seq_length)
            
            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.max_prompt_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            row_dict["input_ids"] = input_ids
            row_dict["attention_mask"] = attention_mask
            row_dict["position_ids"] = position_ids
            row_dict["raw_prompt_ids"] = self.tokenizer.encode(text, add_special_tokens=False)
            batch.append(row_dict)
        # model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
        # input_ids = model_inputs.pop("input_ids")[0]
        # attention_mask = model_inputs.pop("attention_mask")[0]
        # row_dict["multi_modal_inputs"] = dict(model_inputs)
        #print ('batch', batch)
        batch = collate_fn(batch)
        #print ('collate', batch)
        batch = DataProto.from_single_dict(batch)
        #print ('proto', batch)
        gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
        #print ('gen_batch', gen_batch)
        return gen_batch
        #     inputs = inputs.to("cuda")
        #     #print ('input devices', inputs.input_ids.device, self.model.device)
        #     outputs = self.accelerator.unwrap_model(self.model).generate(**inputs,
        #                                 max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature = self.temperature,
        #                                 pad_token_id = self.tokenizer.pad_token_id).cpu()
        #     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        # raw_action = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print (raw_action)
        # actions = [(observation[aid]['bid'], {'action_str': a}) for aid, a in enumerate(raw_action)]
        # #actions = [(observation[aid]['bid'], eval(a.replace('"name', '\'name'))) for aid, a in enumerate(raw_action)]
        # #print (actions)
        # return actions

    def construct_prompt2(self, observation, task, image_path):
        d = {'prompt': (0, observation), 'task': task, 'image_path': image_path}
        # print (d)
        # exit()
        return self.construct_prompt(d)

    def get_log_prob(self, observation, action, image_path, task):
        action = ['<tool_call>\n'+a+'\n</tool_call><|im_end|>' for a in action]
        batch_messages = [self.construct_prompt2(json.loads(ob), task[oi], image_path[oi]) for oi, ob in enumerate(observation)]
        text = self.processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        #print (text[0])
        #print ('action', action[0])
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        assert inputs.input_ids.size(0) == len(action) == 1
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        # print (self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
        # if self.accelerator.is_main_process:
        #     print (action_ids['input_ids'][0])
        #     print (self.tokenizer.convert_ids_to_tokens(action_ids['input_ids'][0]))
        #print (inputs.input_ids.size(), action_ids["input_ids"].size())
        inputs.input_ids = torch.cat([inputs.input_ids, action_ids["input_ids"][:, :-1]], dim = 1)
        inputs.attention_mask = torch.cat([inputs.attention_mask, action_ids["attention_mask"][:, :-1]], dim = 1)
        outputs = self.model(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask)
        #print ('output', outputs.logits.size())
        # # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        # input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        # attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
        #                         dim = 1)
        # outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        # values = None
        # if isinstance(outputs, Tuple):
        #     values, outputs = outputs
        ## TODO: need to check if token shifting is done correctly
        prediction_probs = self.softmax(outputs.logits[:, -action_ids['input_ids'].shape[1]+1:, :])
        #print (prediction_probs.size())
        selected_prediction_probs = torch.take_along_dim(prediction_probs,\
                                                 action_ids["input_ids"][:, 1:].unsqueeze(2), dim=2).squeeze(2)
        #print (selected_prediction_probs.size())
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        # import IPython; IPython.embed(); exit()
        return torch.log(selected_prediction_probs)#*action_ids["attention_mask"]


def draw_point(image: Image.Image, point: list, color=None):
    from copy import deepcopy
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  
        except ValueError:
            color = (255, 0, 0, 128)  
    else:
        color = (255, 0, 0, 128)  
 
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color  # Red with 50% opacity
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')

if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained('/apdcephfs_sh2_300000800/share_300000800/models/Qwen2.5-VL-7B-Instruct')
    print (processor.image_processor.min_pixels, processor.image_processor.max_pixels)
    #exit()
    dummy_image = Image.open('/apdcephfs_gy2/share_302625455/user/kaixinma/gui_output/qwenvl_verl_test1/images/test32/1742982594.931969_0.png')
    print (dummy_image.height, dummy_image.width)
    resized_height, resized_width  = smart_resize(dummy_image.height,
    dummy_image.width,
    factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
    min_pixels=processor.image_processor.min_pixels,
    max_pixels=processor.image_processor.max_pixels,)
    print (resized_height, resized_width)
    # display_image = draw_point(dummy_image, [56, 237], color='green')
    # display_image.save('/cq/share_1603164/user/kaixinma/point1.png')
    # display_image = draw_point(dummy_image, [107, 2], color='green')
    # display_image.save('/cq/share_1603164/user/kaixinma/point2.png')

    agent = Qwen25VLAgent(policy_lm='/apdcephfs_sh2_300000800/share_300000800/models/Qwen2.5-VL-7B-Instruct')
    observation = {'history': [], 'task': 'dummy task', 'image': dummy_image}

    agent.get_action_inputs([observation])
    exit()
    import numpy as np
    tasks = DataProto.from_single_dict({'tasks': np.array(['this is a task', 'this is another task'], dtype=object), 'ids': torch.Tensor([1, 2])})
    #print (tasks)
    a = tasks.non_tensor_batch['tasks'][0]
    #print (a, type(a))
    b = DataProto.from_dict(tasks[0].batch.unsqueeze(0), {k: np.expand_dims(v, 0) for k, v in tasks[0].non_tensor_batch.items()}, tasks[0].meta_info)
    print (b, type(b))
    c = DataProto.concat([tasks, b])
    print (c)
    #l = [tasks[0], tasks]