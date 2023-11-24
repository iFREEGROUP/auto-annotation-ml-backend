# auto-annotation-ml-backend
基于 yolov8 模型的对象检测自动标注，并生成回 coco 数据集格式数据。
主要用于 cvat 平台导入标注。

> 原 cvat 平台自动标注功能基于无服务平台`nuclio` 实现的。此功能要自定义实现 docker 镜像打包。
> 但是由于 python 依赖问题，造成镜像非常的臃肿，并且叠加网络问题，造成部署自定义自动标注插件非常的困难。
> 因此这个工具主要用于曲线救国的方式：线下生成 coco 标注，再导入 cvat 平台。

### 转换 ultralytics(yolov8)

```python

import torch
from safetensors.torch import save_file,save_model
from ultralytics import YOLO

def rename(name: str):
    name = name.replace("model.0.", "net.b1.0.")
    name = name.replace("model.1.", "net.b1.1.")
    name = name.replace("model.2.m.", "net.b2.0.bottleneck.")
    name = name.replace("model.2.", "net.b2.0.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.4.m.", "net.b2.2.bottleneck.")
    name = name.replace("model.4.", "net.b2.2.")
    name = name.replace("model.5.", "net.b3.0.")
    name = name.replace("model.6.m.", "net.b3.1.bottleneck.")
    name = name.replace("model.6.", "net.b3.1.")
    name = name.replace("model.7.", "net.b4.0.")
    name = name.replace("model.8.m.", "net.b4.1.bottleneck.")
    name = name.replace("model.8.", "net.b4.1.")
    name = name.replace("model.9.", "net.b5.0.")
    name = name.replace("model.12.m.", "fpn.n1.bottleneck.")
    name = name.replace("model.12.", "fpn.n1.")
    name = name.replace("model.15.m.", "fpn.n2.bottleneck.")
    name = name.replace("model.15.", "fpn.n2.")
    name = name.replace("model.16.", "fpn.n3.")
    name = name.replace("model.18.m.", "fpn.n4.bottleneck.")
    name = name.replace("model.18.", "fpn.n4.")
    name = name.replace("model.19.", "fpn.n5.")
    name = name.replace("model.21.m.", "fpn.n6.bottleneck.")
    name = name.replace("model.21.", "fpn.n6.")
    name = name.replace("model.22.", "head.")
    return name

if __name__ == '__main__':

    model = YOLO('./yolov8s.pt')
    model =  model.ckpt['model']
    weights = model.state_dict()
    weights = dict(weights.items())
    weights = {rename(k): t for k, t in weights.items()}
    
    for key, value in weights.items():
        print(key)
    save_file(weights, filename=f'yolov8s.safetensors')
```

### 使用方式

```shell
aaml images/path/dir --config-file path/to/config.toml
```
当前目录生成 `instances_default.json`.