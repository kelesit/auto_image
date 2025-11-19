from websocket import WebSocket
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import os
from pathlib import Path
import logging
import requests # 用于文件上传

logger = logging.getLogger(__name__)

class ComfyUIRunner:
    def __init__(self, server_address: str, workflow_path: str, node_mapping: dict):
        """
        初始化 ComfyUI 运行器

        Args:
            server_address (str): ComfyUI 服务器地址，例如 "127.0.0.1:8188"
            workflow_path (str): ComfyUI workflow API 格式的 JSON 文件路径
            node_mapping (dict): 映射了要修改的节点和内容类型
            output_node_id (str): 最终生成图片的节点的ID
        """
        self.server_address = server_address
        # 规范化服务器地址，去掉末尾斜杠
        if not self.server_address.startswith("https") and not self.server_address.startswith("http"):
            self.base_url = f"http://{self.server_address.rstrip('/')}"
        else:
            self.base_url = self.server_address.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.node_mapping = node_mapping
        self.ws = None
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                self.base_workflow = json.load(f)
            logger.info(f"成功加载工作流模板: {workflow_path}")
        except Exception as e:
            logger.error(f"加载工作流文件失败: {workflow_path}, 错误: {e}")
            self.base_workflow = None

    def connect(self):
        """建立 WebSocket 长连接"""
        if self.ws and self.ws.connected:
            return

        try:
            self.ws = WebSocket()
            # 根据服务器地址自动判断使用 ws 还是 wss
            ws_protocol = "wss" if self.server_address.startswith("https") else "ws"
            # 正确处理服务器地址：去掉协议前缀和末尾斜杠
            ws_address = self.server_address.replace("https://", "").replace("http://", "").rstrip("/")
            ws_url = f"{ws_protocol}://{ws_address}/ws?clientId={self.client_id}"
            
            logger.info(f"正在连接到 WebSocket: {ws_url}")
            self.ws.connect(ws_url)
            logger.info("WebSocket 连接成功建立")
        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            self.ws = None
            raise

    def close(self):
        """关闭 WebSocket 连接"""
        if self.ws:
            try:
                self.ws.close()
                logger.info("WebSocket 连接已关闭")
            except Exception as e:
                logger.error(f"关闭 WebSocket 时出错: {e}")
            finally:
                self.ws = None

    def is_server_running(self, timeout: int = 5) -> bool:
        """检查ComfyUI服务器是否正在运行。"""
        try:
            response = requests.get(self.base_url, timeout=timeout)
            if response.status_code == 200:
                logger.debug("ComfyUI 服务器在线。")
                return True
        except requests.exceptions.RequestException:
            logger.warning(f"无法连接到 ComfyUI 服务器 ({self.base_url})。")
        return False

    def upload_image(self, image_path: str) -> str | None:
        """
        上传单个图片到ComfyUI服务器。

        Args:
            image_path (str): 要上传的图片的本地路径。

        Returns:
            成功则返回服务器上的文件名，否则返回None。
        """
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            return None
            
        url = f"{self.base_url}/upload/image"
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f)}
                data = {'overwrite': 'true'}
                response = requests.post(url, files=files, data=data)
                response.raise_for_status()
            
            response_data = response.json()
            filename = response_data.get('name')
            logger.debug(f"成功上传图片 '{image_path}' 到服务器，文件名为: {filename}")
            return filename
        except requests.exceptions.RequestException as e:
            logger.error(f"上传图片 '{image_path}' 失败: {e}")
            return None
        except Exception as e:
            logger.error(f"处理上传响应时出错: {e}")
            return None

    def _queue_prompt(self, prompt: dict, prompt_id: str):
        p = {"prompt": prompt, "client_id": self.client_id, "prompt_id": prompt_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"{self.base_url}/prompt", data=data)
        urllib.request.urlopen(req)

    def _get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"{self.base_url}/view?{url_values}") as response:
            return response.read()

    def _get_history(self, prompt_id: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _get_generated_images(self, ws: WebSocket, prompt_id: str) -> dict:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                    break
            else:
                continue
        
        history = self._get_history(prompt_id)[prompt_id]
        output_images = {}
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = self._get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
                output_images[node_id] = images_output
        return output_images

    def generate_image(self, a_image_path: str, prompt_text: str, output_dir: Path, output_filename: str) -> str | None:
        """
        根据工作流生成单张C图。
        此版本已根据 '换图老子.json' 简化，只接受一张A图和一个提示词。
        """
        if not self.base_workflow:
            logger.error("工作流未加载，无法生成图片。")
            return None

        if not self.ws or not self.ws.connected:
            try:
                self.connect()
            except Exception:
                return None
            
        # 1. 上传A图
        a_image_filename = self.upload_image(a_image_path)
        if not a_image_filename:
            logger.error("A图上传失败，中断生成流程。")
            return None

        # 2. 准备工作流
        prompt_workflow = json.loads(json.dumps(self.base_workflow))
        a_node_id = self.node_mapping.get("a_image_node")
        prompt_node_id = self.node_mapping.get("prompt_node")
        output_node_id = self.node_mapping.get("output_node")

        if not all([a_node_id, prompt_node_id, output_node_id]):
            logger.error("节点映射不完整 (a_image_node, prompt_node, output_node)，请检查配置。")
            return None

        # 检查节点是否存在于工作流中
        if a_node_id not in prompt_workflow or prompt_node_id not in prompt_workflow:
            logger.error(f"节点ID {a_node_id} 或 {prompt_node_id} 不在工作流中，请检查 '换图老子.json' 和 NODE_MAPPING。")
            return None

        prompt_workflow[a_node_id]["inputs"]["image"] = a_image_filename
        prompt_workflow[prompt_node_id]["inputs"]["text"] = prompt_text
        
        # 3. 执行并获取图片
        # ws = WebSocket()
        try:
            # # 根据服务器地址自动判断使用 ws 还是 wss
            # ws_protocol = "wss" if self.server_address.startswith("https") else "ws"
            # # 正确处理服务器地址：去掉协议前缀和末尾斜杠
            # ws_address = self.server_address.replace("https://", "").replace("http://", "").rstrip("/")
            # ws_url = f"{ws_protocol}://{ws_address}/ws?clientId={self.client_id}"
            
            # logger.info(f"正在连接到 WebSocket: {ws_url}")
            # ws.connect(ws_url)
            
            prompt_id = str(uuid.uuid4())
            self._queue_prompt(prompt_workflow, prompt_id)
            images = self._get_generated_images(self.ws, prompt_id)
        except Exception as e:
            logger.error(f"与ComfyUI通信时出错: {e}")
            try:
                self.close()
                self.connect()
            except:
                pass
            return None
        # finally:
        #     ws.close()

        if not images:
            logger.warning("ComfyUI未返回任何图片。")
            return None

        # 4. 保存图片
        output_dir.mkdir(parents=True, exist_ok=True)
        image_list = images.get(output_node_id)

        if image_list:
            try:
                image_data = image_list[0]
                image = Image.open(io.BytesIO(image_data))
                final_path = output_dir / f"{output_filename}.png"
                image.save(final_path)
                logger.info(f"成功保存C图到: {final_path}")
                return str(final_path)
            except Exception as e:
                logger.error(f"保存图片时出错: {e}")
        else:
            logger.error(f"在指定的输出节点 '{output_node_id}' 中未找到图片。")
            logger.warning(f"可用的输出节点: {list(images.keys())}")

        return None

if __name__ == '__main__':
    
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. 定义配置 (根据 '换图老子.json' 更新)
    SERVER_ADDRESS = "https://u145974--7633a383cb7a.westd.seetacloud.com:8443/"
    WORKFLOW_PATH = r"D:\work\auto_image\换图老子.json"
    NODE_MAPPING = {
        "a_image_node": "78",      # 加载图像节点
        "prompt_node": "111",     # 文本提示节点
        "output_node": "60"       # 保存图像节点
    }
    
    # 2. 初始化运行器
    runner = ComfyUIRunner(
        server_address=SERVER_ADDRESS,
        workflow_path=WORKFLOW_PATH,
        node_mapping=NODE_MAPPING
    )

    # 3. 检查服务器状态并执行任务
    if runner.is_server_running():
        logger.info("服务器在线，准备开始生成任务。")
        
        # 定义输入
        a_img_path = r"D:\work\auto_image\2449912965.png" # 请使用您本地的图片路径
        text_prompt = "change the background to a modern living room with light grey carpet, white walls, a beige sofa, and soft natural daylight coming through sheer curtains."
        output_directory = Path("./output_images")
        output_file = "test_generation_01"

        # 检查输入图片是否存在
        if not os.path.exists(a_img_path):
            logger.error(f"请确保输入图片存在: {a_img_path}")
        else:
            generated_path = runner.generate_image(
                a_image_path=a_img_path,
                prompt_text=text_prompt,
                output_dir=output_directory,
                output_filename=output_file
            )
            if generated_path:
                logger.info(f"任务完成，图片已保存到: {generated_path}")
            else:
                logger.error("任务失败，未能生成图片。")

    else:
        logger.error("服务器不在线，请先启动ComfyUI服务器。")
        logger.info("启动命令示例: cd /root/autodl-tmp/ComfyUI && comfy launch")