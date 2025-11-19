import os
from dashscope import MultiModalConversation
import dashscope 



def generate_prompt(image_path):
    """
    使用DashScope的多模态对话模型生成图像描述（Prompt）
    
    Args:
        image_path: 图像的本地路径或URL
        
    Returns:
        生成的图像描述字符串
    """
    if os.path.isfile(image_path):
        image_path = f"file://{image_path}"
    instruction = """Analyze the provided image and generate a precise, effective "change the background to..." instruction for an image editing tool like Flux. Your task is to extract only the key elements of the environment (the background) and create a prompt that accurately recreates this space without mentioning any specific foreground object.

Follow these steps:

Identify the Scene & Style: Determine the overall setting (e.g., living room, bedroom, office, outdoor patio) and its dominant design style (e.g., modern minimalist, Scandinavian, industrial, rustic, contemporary).
Extract Key Background Elements:
Flooring: Describe the material, color, and pattern (e.g., light grey wide-plank wood flooring, concrete floor, carpet).
Walls & Architecture: Note wall colors, materials (e.g., white walls, brick wall), major architectural features (e.g., large windows, bookshelves, fireplace).
Furniture & Fixtures (Background): Identify major background furniture or fixtures (e.g., sofa, bookshelf, desk, bed frame, kitchen cabinets, sink, stove, chairs). Describe their overall form and primary material (e.g., "white armchair", "dark wood bookshelf", "metal-framed desk"), but avoid specific surface textures or colors unless critical.
Lighting: Identify the primary light source (e.g., natural daylight from windows, ceiling lights) and describe the quality of light (e.g., bright, diffused, soft shadows).
Decor & Details: Mention any significant decor elements that define the space (e.g., large abstract painting, potted plant in corner, curtain type/color).
Generate the Prompt: Combine all the information from Step 2 into a single, coherent sentence starting with "change the background to" and ending with a period. Ensure the language is clear and descriptive. DO NOT include any description of the main object in the foreground (e.g., a table, chair, sofa). Instead, imply the space by describing the area where it would be placed (e.g., "a dining area", "a seating area") or simply leave it implied by the context.
Output Format: Only output the generated "change the background to..." sentence.
"""
    messages = [
        {'role':'system',
         'content': instruction},
        {'role':'user',
         'content': [{'image': image_path}]}
    ]
    
    response = MultiModalConversation.call(
        api_key="sk-17d17a164fb44ff99d88647bc6a1d551",
        model='qwen3-vl-plus',
        messages=messages
    )
    
    prompt = response["output"]["choices"][0]["message"].content[0]["text"]
    return prompt

if __name__ == "__main__":
    test_image_path = "/root/auto_image/src/2444117027.png"  # 替换为你的图像路径
    prompt = generate_prompt(test_image_path)
    print("生成的Prompt:", prompt)  
