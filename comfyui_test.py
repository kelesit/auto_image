import websocket
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import os

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt, prompt_id):
    p = {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    urllib.request.urlopen(req).read()

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = str(uuid.uuid4())
    queue_prompt(prompt, prompt_id)
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def save_images(images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for node_id in images:
        for idx, image_data in enumerate(images[node_id]):
            img = Image.open(io.BytesIO(image_data))
            img.save(os.path.join(save_dir, f"{node_id}_{idx}.png"))

# 1. 读取 workflow.json
with open("workflow.json", "r") as f:
    workflow = json.load(f)

# 2. 替换图片和prompt
img1_path = "2443498863.png"
img2_path = "2443714618.png"
prompt_text = "change the background with a curtain and the kitchen"

workflow["191"]["inputs"]["image"] = os.path.basename(img1_path)
workflow["192"]["inputs"]["image"] = os.path.basename(img2_path)
workflow["6"]["inputs"]["text"] = prompt_text

# 3. 连接 websocket
ws = websocket.WebSocket()
ws.connect(f"ws://{server_address}/ws?clientId={client_id}")

# 4. 生成图片
images = get_images(ws, workflow)
ws.close()

# 5. 保存图片
save_images(images, "output_dir")