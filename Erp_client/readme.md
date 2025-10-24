# 负责与ERP 的对接交互


## spu图片上传
POST https://erp.baycheer.cn/api/imageAi/uploadSpuPackageImage?app_id=392016&app_token=NRLGZL9XVMPKIN003PO88OGV9MLL7SPV
上面url是我本地的地址，ip：192.168.100.194，可先本地测试下，没问题再上线。
线上地址是https://erp.baycheer.com/api/imageAi/uploadSpuPackageImage

请求参数：
spu_id: int，spu id
image_id: int，图片id
image: array，图片对象数组，通过上传图片的方式添加
image_url: string，json字符串，通过图片url添加，格式如下，url为图片地址，name为图片名
[
    {
        "url": "https://res.litfad.net/site/img/item/2020/05/16/731718.jpg",
        "name": "731718.jpg"
    }
]

image和image_url只支持一个，image不为空用image，为空才使用image_url

返回：code==0操作成功，否则操作失败
{
    "code": 0,
    "data": [],
    "msg": "上传成功"
}



线上：
上传图片到图片包接口 app_id: 392029, token: LU3FFG1YTI0F7MJ4HMBKDEGTPXMG8YFK