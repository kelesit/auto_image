import httpx
import asyncio
from typing import List, Dict, Optional, Union
from pathlib import Path
import json


class ErpImageClient:
    """ERP 图片上传客户端"""
    
    def __init__(self, app_id: str, app_token: str, base_url: str = "https://erp.baycheer.com", timeout: float = 30.0):
        """
        初始化客户端
        
        Args:
            app_id: 应用ID
            app_token: 应用Token
            base_url: ERP基础URL（默认线上地址）
            timeout: 请求超时时间（秒）
        """
        self.app_id = app_id
        self.app_token = app_token
        self.base_url = base_url
        self.timeout = timeout
        
    def _build_url(self, endpoint: str) -> str:
        """构建完整的API URL"""
        return f"{self.base_url}{endpoint}?app_id={self.app_id}&app_token={self.app_token}"
    
    def upload_by_url(self, spu_id: int, image_id: int, image_urls: List[Dict[str, str]]) -> Dict:
        """
        通过URL上传图片
        
        Args:
            spu_id: SPU ID
            image_id: 图片ID
            image_urls: 图片URL列表，格式: [{"url": "...", "name": "..."}]
        
        Returns:
            响应数据字典，{"code": 0, "msg": "...", "data": [...]}
            
        Example:
            >>> client = ErpImageClient(app_id="392029", app_token="YOUR_TOKEN")
            >>> result = client.upload_by_url(
            ...     spu_id=123456,
            ...     image_id=1,
            ...     image_urls=[{"url": "https://example.com/1.jpg", "name": "1.jpg"}]
            ... )
            >>> print(result)
        """
        url = self._build_url("/api/imageAi/uploadSpuPackageImage")
        
        data = {
            "spu_id": spu_id,
            "image_id": image_id,
            "image_url": json.dumps(image_urls, ensure_ascii=False)
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, data=data)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            return {"code": -1, "msg": f"HTTP错误: {e.response.status_code}", "data": []}
        except httpx.RequestError as e:
            return {"code": -1, "msg": f"请求错误: {str(e)}", "data": []}
        except Exception as e:
            return {"code": -1, "msg": f"未知错误: {str(e)}", "data": []}
    
    def upload_by_file(self, spu_id: int, image_id: int, image_paths: List[Union[str, Path]]) -> Dict:
        """
        通过文件上传图片
        
        Args:
            spu_id: SPU ID
            image_id: 图片ID
            image_paths: 图片文件路径列表
            
        Returns:
            响应数据字典
            
        Example:
            >>> client = ErpImageClient(app_id="392029", app_token="YOUR_TOKEN")
            >>> result = client.upload_by_file(
            ...     spu_id=123456,
            ...     image_id=1,
            ...     image_paths=["./image1.jpg", "./image2.jpg"]
            ... )
        """
        url = self._build_url("/api/imageAi/uploadSpuPackageImage")
        
        data = {
            "spu_id": spu_id,
            "image_id": image_id
        }
        
        files = []
        try:
            for path in image_paths:
                file_path = Path(path)
                if not file_path.exists():
                    return {"code": -1, "msg": f"文件不存在: {path}", "data": []}
                
                files.append(
                    ("image[]", (file_path.name, open(file_path, "rb"), "image/jpeg"))
                )
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, data=data, files=files)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            return {"code": -1, "msg": f"HTTP错误: {e.response.status_code}", "data": []}
        except httpx.RequestError as e:
            return {"code": -1, "msg": f"请求错误: {str(e)}", "data": []}
        except Exception as e:
            return {"code": -1, "msg": f"未知错误: {str(e)}", "data": []}
        finally:
            # 关闭所有打开的文件
            for _, (_, file_obj, _) in files:
                try:
                    file_obj.close()
                except:
                    pass
    
    async def upload_by_url_async(self, spu_id: int, image_id: int, image_urls: List[Dict[str, str]]) -> Dict:
        """
        异步上传图片（通过URL）
        
        Args:
            spu_id: SPU ID
            image_id: 图片ID
            image_urls: 图片URL列表
            
        Returns:
            响应数据字典
        """
        url = self._build_url("/api/imageAi/uploadSpuPackageImage")
        
        data = {
            "spu_id": spu_id,
            "image_id": image_id,
            "image_url": json.dumps(image_urls, ensure_ascii=False)
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, data=data)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            return {"code": -1, "msg": f"HTTP错误: {e.response.status_code}", "data": []}
        except httpx.RequestError as e:
            return {"code": -1, "msg": f"请求错误: {str(e)}", "data": []}
        except Exception as e:
            return {"code": -1, "msg": f"未知错误: {str(e)}", "data": []}
    
    async def batch_upload_async(self, upload_tasks: List[Dict]) -> List[Dict]:
        """
        批量异步上传图片（高性能）
        
        Args:
            upload_tasks: 上传任务列表，格式:
                [
                    {"spu_id": 123, "image_id": 1, "image_urls": [{"url": "...", "name": "..."}]},
                    ...
                ]
        
        Returns:
            结果列表
        """
        tasks = [
            self.upload_by_url_async(
                task["spu_id"],
                task["image_id"],
                task["image_urls"]
            )
            for task in upload_tasks
        ]
        return await asyncio.gather(*tasks)


# 便捷函数
def create_client(env: str = "production") -> ErpImageClient:
    """
    快速创建客户端
    
    Args:
        env: 环境类型，"local" 或 "production"（默认）
        
    Returns:
        ErpImageClient实例
    """
    if env == "local":
        return ErpImageClient(
            app_id="392016",
            app_token="NRLGZL9XVMPKIN003PO88OGV9MLL7SPV",
            base_url="http://192.168.100.194"
        )
    else:
        return ErpImageClient(
            app_id="392029",
            app_token="LU3FFG1YTI0F7MJ4HMBKDEGTPXMG8YFK",
            base_url="https://erp.baycheer.com"
        )


if __name__ == "__main__":
    # 示例：同步上传
    print("=== 测试图片上传 ===")
    client = create_client("production")
    
    # result = client.upload_by_url(
    #     spu_id=123456,
    #     image_id=1,
    #     image_urls=[
    #         {"url": "https://res.litfad.net/site/img/item/2020/05/16/731718.jpg", "name": "731718.jpg"}
    #     ]
    # )
    # print(f"上传结果: {result}")
    
    # # 示例：异步批量上传
    # async def test_batch():
    #     client = create_client("local")
    #     tasks = [
    #         {
    #             "spu_id": 123456,
    #             "image_id": 1,
    #             "image_urls": [{"url": "https://res.litfad.net/site/img/item/2020/05/16/731718.jpg", "name": "731718.jpg"}]
    #         },
    #         {
    #             "spu_id": 123457,
    #             "image_id": 2,
    #             "image_urls": [{"url": "https://res.litfad.net/site/img/item/2020/05/16/731719.jpg", "name": "731719.jpg"}]
    #         }
    #     ]
    #     results = await client.batch_upload_async(tasks)
    #     print(f"\n批量上传结果: {results}")
    
    # print("\n=== 测试批量上传 ===")
    # asyncio.run(test_batch())

    result = client.upload_by_file(
        spu_id=123456,
        image_id=1,
        image_paths=["./731718.jpg", "./731719.jpg"]
    )
    print(f"文件上传结果: {result}")