use std::path::PathBuf;

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Settings {
    #[serde(default)]
    pub model_path: String,

    pub labels:Vec<String>,
}
