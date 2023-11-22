# auto-annotation-ml-backend
基于 yolov8 模型的对象检测自动标注，并生成回 coco 数据集格式数据。
主要用于 cvat 平台导入标注。

> 原 cvat 平台自动标注功能基于无服务平台`nuclio` 实现的。此功能要自定义实现 docker 镜像打包。
> 但是由于 python 依赖问题，造成镜像非常的臃肿，并且叠加网络问题，造成部署自定义自动标注插件非常的困难。
> 因此这个工具主要用于曲线救国的方式：线下生成 coco 标注，再导入 cvat 平台。

### 使用方式

TODO