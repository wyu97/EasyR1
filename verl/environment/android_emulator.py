import os
import shutil
import subprocess, signal
import re
from time import sleep
import random
import time
import click
import warnings

from appium import webdriver
from appium.options.android import UiAutomator2Options

import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from termcolor import colored, cprint
import concurrent.futures
import numpy as np
import traceback
from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import json
import requests
import io
import ray
import psutil 
from openai import AzureOpenAI
import openai

#ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))

def extract_status(text):
    match = re.search(r'Status:\s*(\w+)', text)
    if match:
        return match.group(1)
    else:
        return None

def process_image_gpt(image_path):
    image = Image.open(image_path)
    image = image.resize((image.width // 2, image.height // 2))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return {"url": f"data:image/png;base64,{image_base64}"}

def call_gpt_traj(prompt, image_list):
    client0 = AzureOpenAI(
        azure_endpoint="https://francecentral.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
        api_key="83f30a2a22324395b854bd343db38d85",
        api_version="2024-08-01-preview"
    )
    client1 = AzureOpenAI(
        azure_endpoint="https://it008-wu-001.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
        api_key="1bfd69ea13414a8fa93e50c4d4156dda",
        api_version="2024-08-01-preview"
    )
    client2 = AzureOpenAI(
        azure_endpoint = "https://text-embedding-3-small-ailab.openai.azure.com/", 
        api_key= "b229340d8931472392a45e66eeef05a6",  
        api_version="2024-02-01"
    )
    client3 = AzureOpenAI(
        azure_endpoint = "https://eastus.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview", 
        api_key= "dc910adb4572435495ded4fc3947c07e",  
        api_version="2024-08-01-preview"
    )
    client4 = AzureOpenAI(
        azure_endpoint = "https://it008-gpto1.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview", 
        api_key= "860730aa3a0f497baf199642e2f22d9e",  
        api_version="2024-08-01-preview"
    )
    client5 = AzureOpenAI(
        azure_endpoint = "https://eastus.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview", 
        api_key= "13eef036c63f487ea83e4117cbca05e5",  
        api_version="2024-08-01-preview"
    )

    messages = [{"role": "system", "content": [{"type": "text", "text": "You're an expert evaluator for mobile navigation tasks.\nYou are given a task that a mobile agent is supposed to complete and a series of mobile phone screenshots taken when the agent execute the task. \nYour job is to closely examine the content in the screenshots and determine whether the task is successfully completed.\nYou should pay close attention to the screenshot and make sure that all of the task requirements have been satisfied. For example, if a task requires finding certain information, simply searching for it is not sufficient and the resulting screenshots have to show the required information to be considered successful. \nRespond in this format:\nThought: <a detailed reasoning process about whether the task is successful or not.>\nStatus: success or failure (don't return anything else)"}]}]
    messages.append({"role": "user", "content": [{"type": "text", "text": "Task: "+prompt+'\nScreenshots: '}]})
    for i in range(len(image_list)):
        if os.path.exists(image_list[i]):
            messages[-1]['content'].append({"type": "image_url", "image_url": process_image_gpt(image_list[i])})
        else:
            print ('image does not exist, skipping in eval', image_list[i])

    for idx, client in enumerate([client2, client1, client0, client5, client4, client3]):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
                max_tokens=200,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            print (f"client {idx} Failed to call LLM: " + str(e))
            continue


def call_vllm_traj(prompt, image_list):
    messages = [{"role": "system", "content": [{"type": "text", "text": "You're an expert evaluator for mobile navigation tasks.\nYou are given a task that a mobile agent is supposed to complete and a series of mobile phone screenshots taken when the agent execute the task. \nYour job is to closely examine the content in the screenshots and determine whether the task is successfully completed.\nYou should pay close attention to the screenshot and make sure that all of the task requirements have been satisfied. For example, if a task requires finding certain information, simply searching for it is not sufficient and the resulting screenshots have to show the required information to be considered successful. \nRespond in this format:\nThought: <a detailed reasoning process about whether the task is successful or not.>\nStatus: success or failure (don't return anything else)"}]}]
    messages.append({"role": "user", "content": [{"type": "text", "text": "Task: "+prompt+'\nScreenshots: '}]})
    for i in range(len(image_list)):
        if os.path.exists(image_list[i]):
            messages[-1]['content'].append({"type": "image_url", "image_url": process_image_gpt(image_list[i])})
        else:
            print ('image does not exist, skipping in eval', image_list[i])
    payload = {
            #"model": 'ck',
            "messages": messages,
            "max_tokens": 500,
            "top_p": 0.9,
            "temperature": 0.0
        }
    headers = {
            "Content-Type": "application/json",
        }
    response = requests.post(
        "http://30.159.161.75:8081/v1/chat/completions",
        headers=headers,
        json=payload,
        proxies={"http": None, "https": None}
    )
    # print (response)
    return response.json()['choices'][0]['message']['content']

class Qwen25VLEvaluator:
    def __init__(self):
        pass
        #self.threshold = 0.001 * 255**2

    def __call__(self, last_k_images, intent: str) -> bool:
        """
        last_two_images: a list of two image path. [last_image_path, this_image_path]
        intent: a string representing the user's intent

        Returns:
        - True if the task is completed
        - False otherwise

        If there's an error, it will return False and print the error message
        """
        
        try:
            response_text = call_vllm_traj(intent, last_k_images)
        except:
            try:
                response_text = call_gpt_traj(intent, last_k_images)
            except:
                print ('VLLM service and GPT failed to get response')
                response_text = None

        if extract_status(response_text) is not None and 'success' in extract_status(response_text).lower():
            print("Success!")
            print("task")
            print(intent)
            print("response")
            print(response_text)
            return 1
        return 0

class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7
    Menu=8
    Wait=9

@dataclass
class AndroidAction():
    action_type: ActionType
    touch_point: Tuple[float, float] = None
    lift_point: Tuple[float, float] = None
    typed_text: str = None
    time: float = None

    def __str__(self):
        # Construct the basic action type string.
        components = [f"Action Type: {self.action_type.name}"]

        # Format and add touch_point if it's not None.
        if self.touch_point:
            touch_point_str = f"({self.touch_point[0]:.4f}, {self.touch_point[1]:.4f})"
            components.append(f"Touch Point: {touch_point_str}")

        # Format and add lift_point if it's not None.
        if self.lift_point:
            lift_point_str = f"({self.lift_point[0]:.4f}, {self.lift_point[1]:.4f})"
            components.append(f"Lift Point: {lift_point_str}")

        # Add typed_text if it's not None.
        if self.typed_text:
            components.append(f"Typed Text: '{self.typed_text}'")

        # Join all components into a single string.
        return ", ".join(components)

    def to_act(self):
        pass


def qwen25vl_translate_action(full_output, width=1092, height=2408):
    action = json.loads(full_output.replace("\"\n</tool_call>", "<tool_call>").encode().decode('unicode_escape').split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    try:
        if action['arguments']['action'] == 'click':
            x = float(action['arguments']['coordinate'][0])/width
            y = float(action['arguments']['coordinate'][1])/height
            touch_point = (x, y)
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=touch_point)
        elif action['arguments']['action'] == 'type':
            text = action['arguments']['text']
            return AndroidAction(action_type=ActionType.Type, typed_text=text)
        elif action['arguments']['action'] == 'system_button':
            if action['arguments']['button'] == 'Home':
                return AndroidAction(action_type=ActionType.GoHome)
            elif action['arguments']['button'] == 'Back':
                return AndroidAction(action_type=ActionType.GoBack)
            elif action['arguments']['button'] == 'Enter':
                return AndroidAction(action_type=ActionType.Enter)
            elif action['arguments']['button'] == 'Menu':
                return AndroidAction(action_type=ActionType.Menu)
        elif action['arguments']['action'] == 'terminate':
            if action['arguments']['status'] == 'success':
                return AndroidAction(action_type=ActionType.TaskComplete)
            else:
                return AndroidAction(action_type=ActionType.TaskImpossible)
        elif action['arguments']['action'] == 'swipe':
            x1 = float(action['arguments']['coordinate'][0])/width
            y1 = float(action['arguments']['coordinate'][1])/height
            x2 = float(action['arguments']['coordinate2'][0])/width
            y2 = float(action['arguments']['coordinate2'][1])/height
            touch_point = (x1, y1)
            lift_point = (x2, y2)
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=lift_point)
        elif action['arguments']['action'] == 'wait':
            return AndroidAction(action_type=ActionType.Wait, time=float(action['arguments']['time']))
        elif action['arguments']['action'] == 'answer':
            text = action['arguments']['text']
            return AndroidAction(action_type=ActionType.TaskComplete, typed_text=text)
        # elif action['arguments']['action'] == 'long_press':
        #     x = int(action['arguments']['coordinate'][0]//1092)
        #     y = int(action['arguments']['coordinate'][1]//2268)
        #     touch_point = (x, y)
        #     return AndroidAction(action_type=ActionType.LongPress, touch_point=touch_point, lift_point=touch_point, time=float(action['arguments']['time']))
        else:
            print(f"Action {action} not supported yet.")
            return AndroidAction(action_type=ActionType.Idle)
    except Exception as e:
        print(f"Action {action} Parsing Error: {e}")
        return AndroidAction(action_type=ActionType.Idle)

def escape_shell_text(text):
    # List of characters to escape
    chars_to_escape = ['\\','"', "'", '`', '$']
    
    # Escape the characters by adding a backslash before them
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    text = text.replace(" ", "%s")
    return text

def kill_all_emulators(adb_path, emulators=None):
    subprocess.run([adb_path, 'kill-server'])
    sleep(2)
    subprocess.run([adb_path, 'start-server'])
    sleep(2)
    # Get the list of connected devices
    result = subprocess.run([adb_path, 'devices'], stdout=subprocess.PIPE)
    devices_output = result.stdout.decode('utf-8')
    
    # Find all emulator device names using a regular expression
    running_emulators = re.findall(r'emulator-\d+', devices_output)
    
    # Shut down each emulator found
    for emulator in emulators:
        if emulator not in running_emulators:
            continue
        subprocess.run([adb_path, '-s', emulator, 'emu', 'kill'])
        print(f'{emulator} has been shut down.')

    # Wait for emulators to stop
    sleep(5)

    # Force-kill any lingering emulator processes (Linux/macOS example)
    os.system("pkill -9 -f qemu-system")  # Adjust for Windows if needed

    if not emulators:
        print("No running emulators found.")

    # Verify cleanup
    result = subprocess.run([adb_path, 'devices'], stdout=subprocess.PIPE)
    devices_output = result.stdout.decode('utf-8')
    remaining = re.findall(r'emulator-\d+', devices_output)
    if remaining:
        print(f"Warning: Emulators still detected after cleanup: {remaining}")
    else:
        print("All emulators successfully shut down.")

def clone_avd(src_avd_name, tar_avd_name, android_avd_home):
    """
    Clone the source AVD to the target AVD.

    Parameters:
    - src_avd_name: The name of the source AVD folder.
    - tar_avd_name: The name of the target AVD folder.
    - android_avd_home: The path to the .android/avd directory.

    This function copies the source AVD folder and its .ini file to a new target AVD
    and updates the paths inside the .ini files accordingly.
    """

    # Paths for source and target AVD directories and .ini files
    src_avd_dir = os.path.join(android_avd_home, src_avd_name + '.avd')
    tar_avd_dir = os.path.join(android_avd_home, tar_avd_name + '.avd')
    src_ini_file = os.path.join(android_avd_home, src_avd_name + '.ini')
    tar_ini_file = os.path.join(android_avd_home, tar_avd_name + '.ini')

    # Copy the AVD folder
    colorful_print(f"Copying the AVD folder from {src_avd_dir} to {tar_avd_dir}", "green")
    if not os.path.exists(tar_avd_dir):
        shutil.copytree(src_avd_dir, tar_avd_dir)

    # Copy the .ini file and modify it for the new AVD
    with open(src_ini_file, 'r') as src_ini, open(tar_ini_file, 'w') as tar_ini:
        for line in src_ini:
            tar_ini.write(line.replace(src_avd_name, tar_avd_name))

    # Update paths inside the target AVD's .ini files
    for ini_name in ['config.ini', 'hardware-qemu.ini']:
        ini_path = os.path.join(tar_avd_dir, ini_name)
        if os.path.exists(ini_path):
            with open(ini_path, 'r') as file:
                lines = file.readlines()
            with open(ini_path, 'w') as file:
                for line in lines:
                    # Update paths and AVD name/ID
                    new_line = line.replace(src_avd_name, tar_avd_name)
                    file.write(new_line)

    # Update the snapshots' hardware.ini file if it exists
    snapshots_hw_ini = os.path.join(tar_avd_dir, 'snapshots', 'default_boot', 'hardware.ini')
    if os.path.exists(snapshots_hw_ini):
        with open(snapshots_hw_ini, 'r') as file:
            lines = file.readlines()
        with open(snapshots_hw_ini, 'w') as file:
            for line in lines:
                # Update AVD name/ID
                new_line = line.replace(src_avd_name, tar_avd_name)
                file.write(new_line)

@ray.remote
class AndroidEmulator():
    def __init__(self, avd_name, max_steps, temp_path, evaluator, emulator_path="~/Android/Sdk/emulator/emulator", appium_server_url='http://localhost:4723', no_window=False, udid = None, task = None, image_size = None, save_images = False, record=False):
        """
        temp_path temporary path to store the images for evaluation
        """
        self.temp_path = temp_path
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.emulator_path = os.path.expanduser(emulator_path)
        self.avd_name = avd_name
        self.save_images = save_images
        self.image_id = udid + '_' + str(time.time())
        port_number = udid.split("-")[-1]
        self.udid = udid
        cprint(colored(f"Starting the Emulator", "green"))
        command = f"""{self.emulator_path} -avd {self.avd_name} "-no-audio" "-skip-adb-auth" "-no-boot-anim" "-gpu" "auto" "-no-snapshot-save" -port {port_number} -http-proxy http://9.21.0.122:11113 -dns-server 8.8.8.8 -memory 3072 -cores 2 -verbose"""
        if no_window:
            command += " -no-window"
        log_file = f"emulator_log/emulator_{port_number}_log.txt"
        print(f"executing command {command}")
        #self.emulator_process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.emulator_process = subprocess.Popen(command, shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
        #sleep(30)
        if not self.wait_for_emulator(port_number, timeout=60):
            raise Exception(f"Emulator {self.udid} failed to boot within 60 seconds")
        subprocess.run(["adb", "-s", f"emulator-{port_number}", "shell", "settings", "put", "global", "http_proxy", "9.21.0.122:11113"])
        self.record = record
        if self.record:
            self.record_random_id = random.randint(0, 100000)
            try_record_command = f"""adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_1.mp4"""
            # redirect the output and error to the output of the main process
            import sys
            print(f"Trying to record the screen of {self.udid}")
            self.try_record_process = subprocess.Popen(try_record_command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
            sleep(20)
            self.try_record_process.terminate()
            try:
                self.try_record_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.try_record_process.kill()
                self.try_record_process.wait()
            sleep(5)
            print(f"Recording the screen of {self.udid}")
            do_record_command = f"""adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_1.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_2.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_3.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_4.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_5.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_6.mp4"""
            self.record_process = subprocess.Popen(do_record_command, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid) # should be good the second time
            sleep(5)

        capabilities = dict(
            platformName='Android',
            automationName='uiautomator2',
            deviceName='Android',
            newCommandTimeout="120000",
            adbExecTimeout="120000",
            uiautomator2ServerInstallTimeout="120000",
            uiautomator2ServerLaunchTimeout="120000",
            uiautomator2ServerReadTimeout="120000",
            noSign=True
        )
        if udid:
            capabilities["udid"] = udid
        self.options = UiAutomator2Options().load_capabilities(capabilities)
        self.appium_server_url = appium_server_url
        print ('Trying appium server at ', self.appium_server_url)
        for i in range(3):
            try:
                self.driver = webdriver.Remote(self.appium_server_url, options=self.options)
                print("connected!")
                break
            except Exception as e:
                cprint(colored(f"Failed to connect to the appium server: {e}\n Retrying", "red"))
                if i == 3:
                    raise Exception("Failed to connect to the appium server")
                sleep(20)
        self.terminated = False
        self.max_steps = max_steps
        self.steps = 0
        screen_size = self.driver.get_window_size()
        self.screen_size = (screen_size["width"], screen_size["height"])
        self.current_task = task

        self.image_size = image_size
        print ('Current image size is set to', self.image_size)
        self.history = []
        self.evaluator = evaluator

    def wait_for_emulator(self, port, timeout=60):
        """Wait until emulator is fully booted."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = subprocess.run(["adb", "-s", f"emulator-{port}", "shell", "getprop", "sys.boot_completed"], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.stdout.decode().strip() == "1":
                print(f"Emulator {self.udid} booted successfully")
                return True
            sleep(5)
        print(f"Emulator {self.udid} failed to boot")
        return False
    
    def terminate(self):
        
        if self.record:
            # send sigterm to the record process
            os.killpg(os.getpgid(self.record_process.pid), signal.SIGINT)
            sleep(5)
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_1.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_2.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_3.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_4.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_5.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_6.mp4 {self.temp_path}")
            print("it's okay if you see errros like failed to stat remote object '/sdcard/video_1718747809.256034_{i}.mp4' where i is larger than 1.")

        sleep(5)
        self.emulator_process.terminate()
        try:
            self.emulator_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.emulator_process.kill()
            self.emulator_process.wait()
        self.terminated = True
    
    def refresh_driver(self):
        self.driver.quit()
        self.driver = webdriver.Remote(self.appium_server_url, options=self.options)
    
    def count_white_pixels(self, img):
        # Convert the image to RGB format if it's not
        img = img.convert('RGB')
        # Convert image to numpy array
        data = np.array(img)
        # Count white pixels
        # Assuming 'white' is (255, 255, 255)
        white_count = np.sum(np.all(data > 240, axis=-1))
        return white_count > 2_300_000

    def reset(self, task):
        self.history = []
        self.steps = 0
        self.terminated = False
        self.current_task = task
    
    def get_obs(self):
        failed = 0
        for _ in range(3):
            try:
                is_white = True
                for _ in range(5):
                    if not is_white:
                        break
                    sleep(5)
                    screenshot_str = self.driver.get_screenshot_as_base64()
                    imgdata = base64.b64decode(screenshot_str)
                    image =  Image.open(BytesIO(imgdata))
                    is_white = self.count_white_pixels(image)
                # print("Saving observation!")
                image.save(os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png"))
                # Assuming 'image' is your PIL Image object in RGBA mode
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                # colorful_print(f"history: {self.history}", "green")
                # colorful_print(f"prompt: {self.prepare_prompt(self.current_task, self.history)}", "yellow")
                return {"history": self.history,
                        "image": image,
                        "task": self.current_task,
                        "image_path": os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png"),
                        "video_path": os.path.join(self.temp_path, f"video_{self.record_random_id}.mp4") if self.record else None
                }
            except Exception as e:
                print(f"Exception happened during screenshotting")
                print(e)
                print(traceback.format_exc())
                sleep(6)
                if failed == 2:
                    print(f"Failed to get observation: {e}")
                    return None
                failed += 1
                continue
    def get_obs_with_action(self, action=None, screenshot=None, task=None):
        """Get observation with action visualization on the screenshot.
        Args:
            action: AndroidAction object containing the action information
            screenshot: The current screenshot observation
        Returns:
            Same as get_obs() but with an additional image_path_annotated field
        """
        if action is None or screenshot is None:
            return None
        
        # Create a copy of the image for annotation
        if self.steps > 0:
            previous_image_path = os.path.join(self.temp_path, f"{self.image_id}_{self.steps-1}.png")
            image = Image.open(previous_image_path)
            draw = ImageDraw.Draw(image)
        else: 
            return None
        
        try:
            # Try to use DejaVuSans which is commonly available on Linux systems
            font = ImageFont.truetype("DejaVuSans.ttf", 36)
        except IOError:
            try:
                # Try to use a system font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            except IOError:
                # Fall back to default font
                font = ImageFont.load_default()

        if self.steps==1:
            if task is not None:
                font = ImageFont.load_default()
                # Draw the task text at the top-left corner
                draw.text((10, 10), f"Task: {task}", fill='red', font=font)
            annotated_path = os.path.join(self.temp_path, f"{self.image_id}_{self.steps-1}_annotated.png")
            image.save(annotated_path)
        
        # Add action visualization based on action type
        if action.action_type == ActionType.DualPoint:
            # Draw touch and lift points
            touch_x = int(action.touch_point[0] * self.screen_size[0])
            touch_y = int(action.touch_point[1] * self.screen_size[1])
            lift_x = int(action.lift_point[0] * self.screen_size[0])
            lift_y = int(action.lift_point[1] * self.screen_size[1])
            
            # Draw red dot for touch point
            draw.ellipse([touch_x-10, touch_y-10, touch_x+10, touch_y+10], fill='red')
            
            # If it's a swipe, draw the path
            if (touch_x - lift_x)**2 + (touch_y - lift_y)**2 > 10:
                draw.line([touch_x, touch_y, lift_x, lift_y], fill='red', width=3)
                draw.ellipse([lift_x-10, lift_y-10, lift_x+10, lift_y+10], fill='blue')
                
        elif action.action_type == ActionType.Type:
            # Add text annotation for typing action
            draw.text((10, 10), f"Typed: {action.typed_text}", fill='red', font=font)
            
        elif action.action_type in [ActionType.GoBack, ActionType.GoHome, ActionType.Enter, ActionType.Menu]:
            # Add text annotation for system button actions
            draw.text((10, 10), f"Action: {action.action_type.name}", fill='red', font=font)
            
        elif action.action_type == ActionType.Wait:
            # Add text annotation for wait action
            draw.text((10, 10), f"Wait: {action.time}s", fill='red', font=font)
        
        elif action.action_type == ActionType.Enter:
            draw.text((10, 10), f"Enter", fill='red', font=font)
        elif action.action_type == ActionType.TaskComplete:
            # Add text annotation for task complete action
            draw.text((10, 10), f"Action: {action.action_type.name}", fill='red', font=font)

        elif action.action_type == ActionType.TaskImpossible:
            # Add text annotation for task impossible action
            draw.text((10, 10), f"Action: {action.action_type.name}", fill='red', font=font)

        elif action.action_type == ActionType.Idle:
            # Add text annotation for idle action
            draw.text((10, 10), f"Action: {action.action_type.name}", fill='red', font=font)
        else:
            pass
            
        # Save the annotated image
        annotated_path = os.path.join(self.temp_path, f"{self.image_id}_{self.steps}_annotated.png")
        image.save(annotated_path)
        
        # Add the annotated image path to the observation
        screenshot["image_path_annotated"] = annotated_path
        return screenshot

    def step(self, raw_action: str, skip_record=False):
        if self.terminated:
            return None
        try:
            # colorful_print(f"raw action: {raw_action}", "green")
            action = qwen25vl_translate_action(raw_action, self.image_size[0], self.image_size[1])
            # colorful_print(f"translated action: {action}", "green")
        except Exception as e:
            print(e)
            print(f"Failed to translate action: {raw_action}, terminating the environment")
            action = AndroidAction(action_type=ActionType.TaskImpossible)
        if not skip_record:
            self.history.append(raw_action)
            self.steps += 1
        if self.steps > self.max_steps:
            action = AndroidAction(action_type=ActionType.TaskImpossible)
            cprint(colored(f"Terminate the Emulator: Max Steps Exceeded {self.max_steps}.", "red"))
        screenshot = None
        info = {}
        for i in range(2):
            try:
                if action.action_type == ActionType.DualPoint:
                    assert len(action.touch_point) == 2
                    assert len(action.lift_point) == 2
                    touch_x = action.touch_point[0] * self.screen_size[0]
                    touch_y = action.touch_point[1] * self.screen_size[1]
                    lift_x = action.lift_point[0] * self.screen_size[0]
                    lift_y = action.lift_point[1] * self.screen_size[1]
                    #print (action)
                    #print ('touch_x, touch_y, lift_x, lift_y', touch_x, touch_y, lift_x, lift_y)
                    if (touch_x - lift_x)**2 + (touch_y - lift_y)**2 < 10:
                        self.driver.tap([(touch_x, touch_y)])
                        #print ('performing a tap')
                    else:
                        self.driver.swipe(touch_x, touch_y, lift_x, lift_y)
                        #print ('performing a swipe')
                elif action.action_type == ActionType.Type:
                    # This doesn't work well because of active element
                    for i in range(2):
                        try:
                            sleep(4)
                            element = self.driver.switch_to.active_element
                            element.send_keys(action.typed_text)
                            break
                        except Exception as e:
                            cprint(f"The element is not loaded yet or agent did not click anything", "red")
                    
                elif action.action_type == ActionType.GoBack:
                    self.driver.back()
                elif action.action_type == ActionType.GoHome:
                    self.driver.press_keycode(3)
                elif action.action_type == ActionType.Enter:
                    self.driver.press_keycode(66)
                elif action.action_type == ActionType.TaskComplete:
                    self.terminated = True
                elif action.action_type == ActionType.TaskImpossible:
                    self.terminated = True
                elif action.action_type == ActionType.Wait:
                    sleep(action.time)
                elif action.action_type == ActionType.Idle:
                    pass
                else:
                    raise Exception(f"Unknown action type: {action.action_type}")
                action_success = True
                # Get both regular and annotated screenshots
                screenshot = self.get_obs()
                screenshot_with_action = self.get_obs_with_action(action, screenshot, self.current_task)
                
                break
            except Exception as e:
                print("an Exception occurred during environment interaction: ", e)
                print("Retrying")
                sleep(10)
                if i == 1:
                    action_success = False
                    info["error"] = str(e)
                    self.driver.quit()
                    self.terminate()
                    return None
                continue
        r = 0
        if screenshot is not None and self.evaluator is not None and action.action_type == ActionType.TaskComplete:
            r = self.evaluator([os.path.join(self.temp_path, f"{self.image_id}_{self.steps-2}.png"),
                                os.path.join(self.temp_path, f"{self.image_id}_{self.steps-1}.png"), 
                                os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png")], self.current_task)
        info["action_success"] = action_success
        #terminate the environment if there is a success
        if screenshot is None:
            self.terminated = True
        if r >= 1 or self.terminated:
            try:
                self.driver.quit()
                self.terminate()
            except:
                print ('Failed to quit the driver and terminate the emulator, relying on next env reset for clean up')
        if self.terminated and not self.save_images:
            os.system(f"rm -rf {self.temp_path}/*")
        return screenshot, r, self.terminated

def is_port_in_use(port):
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def free_port(port):
    """Kill process using a specific port (Linux/macOS)."""
    import os
    pid = os.popen(f"lsof -t -i:{port}").read().strip()
    if pid:
        os.system(f"kill -9 {pid}")

class BatchedAndroidEnv():
    """
    This class wraps around the android emulator and provides a more infrastructure for free-form GUI navigation
    This is a batched version for Android Env
    cache_avd is the avd to be used the avd is the initial one
    """
    def __init__(self, 
        avd_name, 
        cache_avd_names,
        udids,
        appium_base_port,
        android_avd_home: str = '/nfs/kun2/users/yifei/openended/.android/android_avd/avd',
        emulator_path: str = '~/Android/Sdk/emulator/emulator',
        adb_path: str = "~/Library/Android/sdk/platform-tools/adb",
        run_headless: bool = False,
        max_steps: int = 10,
        evaluators = None,
        temp_path = "/nfs/kun2/users/yifei/openended/logs/images",
        save_images = False,
        all_tasks = None,
        image_size = None,
        record = False):
        
        self.android_avd_home = os.path.expanduser(android_avd_home)
        self.emulator_path = os.path.expanduser(emulator_path)
        self.adb_path = os.path.expanduser(adb_path)
        self.avd_name = avd_name
        self.save_images = save_images
        self.bsize = len(cache_avd_names)
        self.cache_avd_names = cache_avd_names
        self.run_headless = run_headless
        self.max_steps = max_steps
        self.emulator_group_offset = 0

        self.record = record
        self.image_size = image_size
        #self.all_tasks = all_tasks
        self.temp_path = temp_path
        if evaluators is None:
            evaluators = [None for _ in range(self.bsize)]
        self.evaluators = evaluators
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.udids = udids
        self.base_port = appium_base_port
        self.appium_processes = []

        # # Start the appium servers
        # for i in range(self.base_port, self.base_port+self.bsize):
        #     self.appium_processes.append(subprocess.Popen(f"appium --relaxed-security -p {i} > /dev/null", stdout=subprocess.DEVNULL, shell=True))
        #     print("starting appium server at port ", i)
        # self.appium_server_urls = [f"http://127.0.0.1:{i}" for i in range(self.base_port, self.base_port+self.bsize)]
    
    def reset_appium(self):
        for p in self.appium_processes:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        # os.system("pkill -f appium")
        # sleep(2)
        # Force-kill any lingering Appium processes (cross-platform)
        for proc in psutil.process_iter(['pid', 'name']):
            if 'appium' in proc.info['name'].lower() or 'node' in proc.info['name'].lower():
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        sleep(2)
        # Check and free ports if needed
        for port in range(self.base_port, self.base_port + self.bsize):
            if is_port_in_use(port):
                print(f"Port {port} still in use, attempting to free...")
                free_port(port)  # Implement this function (see below)
        #self.base_port = self.base_port + self.bsize * 2
        self.appium_processes = []
        for i in range(self.base_port, self.base_port+self.bsize):
            self.appium_processes.append(subprocess.Popen(f"appium --relaxed-security -p {i} > /dev/null", stdout=subprocess.DEVNULL, shell=True))
            print(f"starting appium server at port {i}")
        # sleep(10)
        self.appium_server_urls = [f"http://localhost:{i}" for i in range(self.base_port, self.base_port+self.bsize)]

    def reset(self, all_tasks):
        """
        Reset the emulator to a clean state
        """
        # if self.appium_processes:
        #     print ('performing fake reset, only returning to Home screen')
        #     go_home_action = "<tool_call>\n{\"arguments\": {\"action\": \"system_button\", \"button\": \"Back\"}}\n</tool_call>"
        #     #all_go_home = [go_home_action]*len(self.emulators)
        #     obj_refs = [emulator.reset.remote(task) for emulator, task in zip(self.emulators, all_tasks)]
        #     _ = ray.get(obj_refs)
        #     obs_refs = [emulator.step.remote(go_home_action, True) for emulator in self.emulators]
        #     results = ray.get(obs_refs)
        #     results = [r[0] for r in results]
        #     return results
        self.all_tasks = all_tasks
        self.reset_appium()
        # If the emulator is already running, kill it,
        # Then delete the cache AVD
        kill_all_emulators(self.adb_path, emulators=self.udids)
        if hasattr(self, "emulator_process"):
            self.emulator_process.send_signal(signal.SIGINT)
            self.emulator_process.wait()
        self.emulators = []
        #sleep(15)
        for cache_avd_name in self.cache_avd_names:
            # print(cache_avd_name)
            for _ in range(3):
                try:
                    cache_avd_path = os.path.join(self.android_avd_home, cache_avd_name + ".avd")
                    cache_avd_ini_path = os.path.join(self.android_avd_home, cache_avd_name + ".ini")
                    # if os.path.exists(cache_avd_path):
                    #     shutil.rmtree(cache_avd_path, ignore_errors=True)
                    # if os.path.exists(cache_avd_ini_path):
                    #     os.remove(cache_avd_ini_path)
                    # sleep(2)
                    if os.path.exists(cache_avd_path) and os.path.exists(cache_avd_ini_path):
                        # KM: if the snapshots exist we don't do anything to speed up
                        continue
                    # Clone the source AVD and start the emulator
                    clone_avd(self.avd_name, cache_avd_name, self.android_avd_home)
                    break
                except OSError as e:
                    print(f"Failed to reset the emulator: {e}")
                    import traceback
                    print(traceback.format_exc())
                    sleep(20)

        def emulator_constructor(udid, appium_server_url, cache_avd_name, evaluator, task):
            return AndroidEmulator(avd_name=cache_avd_name, max_steps=self.max_steps, emulator_path=self.emulator_path, 
                appium_server_url=appium_server_url, 
                no_window=self.run_headless, 
                udid = udid,
                evaluator = evaluator,
                temp_path = os.path.join(self.temp_path, cache_avd_name),
                save_images = self.save_images,
                task=task,
                image_size=self.image_size,
                record=self.record)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     jobs = [executor.submit(emulator_constructor, udid, appium_server_url, cache_avd_name, evaluator, task)
        #         for udid, appium_server_url, cache_avd_name, evaluator, task in 
        #         zip(self.udids, self.appium_server_urls, self.cache_avd_names, self.evaluators, self.all_tasks)]
        #     self.emulators = [job.result() for job in jobs]
        self.emulators = [
            AndroidEmulator.remote(avd_name=cache_avd_name, max_steps=self.max_steps, emulator_path=self.emulator_path, no_window=self.run_headless, udid=udid, appium_server_url=appium_server_url, evaluator=evaluator, temp_path = os.path.join(self.temp_path, cache_avd_name), save_images = self.save_images, task=task, image_size=self.image_size, record=self.record)
            for udid, appium_server_url, cache_avd_name, evaluator, task in zip(
                self.udids, self.appium_server_urls, self.cache_avd_names, self.evaluators, self.all_tasks)
        ]

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     jobs = [executor.submit(emulator.get_obs) for emulator in self.emulators]
        #     # for i, job in enumerate(jobs):
        #         # colorful_print(f"Getting observation from emulator {i}: {job.result()}", "green")
        #     return [job.result() for job in jobs]

        obs_refs = [emulator.get_obs.remote() for emulator in self.emulators]
        results = ray.get(obs_refs)
        return results

    def step(self, actions):
        if not self.emulators:
            raise Exception("Please call reset() before calling step()")
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     jobs = [executor.submit(emulator.step, action) 
        #             for emulator, action in 
        #             zip(self.emulators, actions)]
        #     results = [job.result() for job in jobs]
        obs_refs = [emulator.step.remote(action) for emulator, action in zip(self.emulators, actions)]
        results = ray.get(obs_refs)
        return results

    # def terminate(self):
    #     obs_refs = [emulator.terminate.remote() for emulator in self.emulators]
    #     results = ray.get(obs_refs)
    #     return results

if __name__ == '__main__':
    base_port = 5556
    evaluators = [Qwen25VLEvaluator() for _ in range(2)]
    # env = BatchedAndroidEnv(avd_name="AndroidWorldAvd", 
    #         cache_avd_names=[f"test{i}" for i in range(1,3)], 
    #         android_avd_home='/root/android/avd',
    #         emulator_path='/root/android/emulator/emulator', 
    #         adb_path='/root/android/platform-tools/adb', 
    #         udids = [f"emulator-{base_port+2*i}" for i in range(2)],
    #         max_steps=10, # will have 1 dangling step after stop signal is triggered
    #         appium_base_port = base_port+1198,
    #         run_headless=True, 
    #         evaluators=evaluators,
    #         temp_path = os.path.join('/root/tmp_test/', "images"),
    #         save_images=True,
    #         all_tasks=None,
    #         record=False,
    #         image_size=[532, 1204]
    #     )
    # obs = env.reset(['open up google map', 'open the youtube app'])
    # print (obs)
    # obs = env.reset(['switch to alarm', 'play the music on the first page'])
    # print ('second reset', obs)
    # # env.terminate()

    go_home_action = "<tool_call>\n{\"arguments\": {\"action\": \"system_button\", \"button\": \"Back\"}}\n</tool_call>"
    action = qwen25vl_translate_action(go_home_action)
    print (action)
    
    image_list = ['/cq/share_1603164/user/kaixinma/network_debug1/images/test16/1740648786.143217_4.png', 
                  '/cq/share_1603164/user/kaixinma/network_debug1/images/test16/1740648786.143217_5.png',
                  '/cq/share_1603164/user/kaixinma/network_debug1/images/test16/1740648786.143217_6.png']
    res = evaluators[0](image_list, 'find the current time in Moscow')