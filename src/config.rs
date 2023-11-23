use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Settings {
    pub cvat_host:String,
    pub yolo: YoloConf,
}

#[derive(Deserialize, Debug, Clone)]
pub struct YoloConf {
    pub labels:Vec<String>,
    pub model_path:String,
    pub confidence_threshold:f32,
    pub nms_threshold:f32,
    pub weight_size:String,
}