use axum::{ response::IntoResponse, Json, extract::State };
use hyper::{ StatusCode, header };
use reqwest::Url;
use serde::{ Serialize, Deserialize };
use serde_json::{ Value, json };
use uuid::Uuid;

use crate::{ yolo::inference::Inference, config::Settings };

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Params {
    #[serde(rename = "context")]
    pub context: Option<Value>,

    #[serde(rename = "login")]
    pub login: Option<String>,

    #[serde(rename = "password")]
    pub password: Option<String>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Data {
    #[serde(rename = "image")]
    pub image: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Meta {}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tasks {
    #[serde(rename = "annotations")]
    pub annotations: Option<Vec<String>>,

    #[serde(rename = "cancelled_annotations")]
    pub cancelled_annotations: Option<i32>,

    #[serde(rename = "comment_authors")]
    pub comment_authors: Option<Vec<String>>,

    #[serde(rename = "comment_count")]
    pub comment_count: Option<i32>,

    #[serde(rename = "created_at")]
    pub created_at: Option<String>,

    #[serde(rename = "data")]
    pub data: Data,

    #[serde(rename = "file_upload")]
    pub file_upload: Option<i32>,

    #[serde(rename = "id")]
    pub id: Option<i32>,

    #[serde(rename = "inner_id")]
    pub inner_id: Option<i32>,

    #[serde(rename = "is_labeled")]
    pub is_labeled: Option<bool>,

    #[serde(rename = "last_comment_updated_at")]
    pub last_comment_updated_at: Option<String>,

    #[serde(rename = "meta")]
    pub meta: Option<Meta>,

    #[serde(rename = "overlap")]
    pub overlap: Option<i32>,

    #[serde(rename = "predictions")]
    pub predictions: Option<Vec<String>>,

    #[serde(rename = "project")]
    pub project: Option<i32>,

    #[serde(rename = "total_annotations")]
    pub total_annotations: Option<i32>,

    #[serde(rename = "total_predictions")]
    pub total_predictions: Option<i32>,

    #[serde(rename = "unresolved_comment_count")]
    pub unresolved_comment_count: Option<i32>,

    #[serde(rename = "updated_at")]
    pub updated_at: Option<String>,

    #[serde(rename = "updated_by")]
    pub updated_by: Option<String>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictTask {
    #[serde(rename = "label_config")]
    pub label_config: Option<String>,

    #[serde(rename = "model_version")]
    pub model_version: Option<String>,

    #[serde(rename = "params")]
    pub params: Option<Params>,

    #[serde(rename = "project")]
    pub project: Option<String>,

    #[serde(rename = "tasks")]
    pub tasks: Vec<Tasks>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ValueItem {
    #[serde(rename = "height")]
    pub height: f64,

    #[serde(rename = "rectanglelabels")]
    pub rectanglelabels: Vec<String>,

    #[serde(rename = "rotation")]
    pub rotation: i32,

    #[serde(rename = "width")]
    pub width: f64,

    #[serde(rename = "x")]
    pub x: f64,

    #[serde(rename = "y")]
    pub y: f64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Prediction {
    #[serde(rename = "from_name")]
    pub from_name: String,

    #[serde(rename = "id")]
    pub id: String,

    #[serde(rename = "to_name")]
    pub to_name: String,

    #[serde(rename = "type")]
    pub root_type: String,

    #[serde(rename = "value")]
    pub value: ValueItem,

    pub score: f32,
}

///Predict tasks
///
/// Example request:
/// request = {
///     'tasks': tasks,
///     'model_version': model_version,
///     'project': '{project.id}.{int(project.created_at.timestamp())}',
///     'label_config': project.label_config,
///     'params': {
///         'login': project.task_data_login,
///         'password': project.task_data_password,
///         'context': context,
///     },
/// }
/// tasks see data/raw_task.json
/// @return:
/// Predictions in LS format
pub(crate) async fn post(
    State(infer): State<Inference>,
    State(ref conf): State<Settings>,
    Json(data): Json<PredictTask>
    // body:String
) -> impl IntoResponse {
    let image_path = Url::parse(conf.label_studio_host.as_str()).unwrap();
    let url = image_path.join(data.tasks[0].data.image.as_str()).unwrap();

    let mut headers = header::HeaderMap::new();
    let token = format!("Token {}", conf.access_token);
    headers.insert("Authorization", header::HeaderValue::from_str(&token).unwrap());
    let client = reqwest::ClientBuilder::new().default_headers(headers).build().unwrap();
    let image_byte = client.get(url).send().await.unwrap().bytes().await.unwrap();

    let img_data = Vec::from(image_byte);

    let image = image::load_from_memory_with_format(&img_data, image::ImageFormat::Jpeg).unwrap();
    let result = infer.predict(&image).unwrap();

    let mut predictions = vec![];

    let img_height = image.height();
    let img_width = image.width();

    for (index, label) in result.iter().enumerate() {
        for bbox in label {
            let id = Uuid::new_v4();
            let val = ValueItem {
                height: (bbox.ymax - bbox.ymin) as f64,
                rectanglelabels: vec![conf.labels[index].clone()],
                rotation: 0,
                width: (bbox.xmax - bbox.xmin) as f64,
                x: bbox.xmin as f64,
                y: bbox.ymin as f64,
            };
            let prediction = Prediction {
                from_name: "label".to_owned(),
                id: id.to_string(),
                to_name: "image".to_owned(),
                root_type: "rectanglelabels".to_owned(),
                value: val,
                score: bbox.confidence,
            };
            predictions.push(prediction);
            all_scores.push(bbox.confidence);
        }
    }

    
    (StatusCode::OK, data);
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SetupBody {
    pub(crate) project: String,
    pub(crate) schema: String,
    pub(crate) hostname: String,
    pub(crate) access_token: String,
    pub(crate) model_version: Option<String>,
}

/// {
/// "project": "1.1700543158",
/// "schema": "<View>\n <Image name=\"image\" value=\"$image\"/>\n <RectangleLabels name=\"label\" toName=\"image\">\n \n \n <Label value=\"empty\" background=\"#FFA39E\"/><Label value=\"object\" background=\"#0d63d3\"/></RectangleLabels>\n</View>",
/// "hostname": "http://localhost:8080",
/// "access_token": "4d4a445ce57cfb8f34acd56415510f603ca5ad0e",
/// "model_version": ""
/// }
pub(crate) async fn setup(Json(_body): Json<SetupBody>) -> impl IntoResponse {
    (StatusCode::OK, json!({"model_version":format!("sheld.1.0.0")}).to_string())
}
