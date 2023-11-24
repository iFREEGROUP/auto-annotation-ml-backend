use config::{ ConfigError, Config, Format, FileFormat, Source };
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Settings {
    // pub cvat_host: String,
    pub yolo: YoloConf,
}

#[derive(Deserialize, Debug, Clone)]
pub struct YoloConf {
    pub labels: Vec<String>,
    pub model_path: String,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub weight_size: String,
}

#[derive(Debug, Clone)]
struct MemorySource<F: Format> {
    content: String,
    format: F,
}

impl MemorySource<FileFormat> {
    pub fn new() -> Self {
        let contents = include_str!("../example_config.toml");
        Self {
            content: contents.to_string(),
            format: FileFormat::Toml,
        }
    }
}

impl<F: Format + Clone + Send + Sync + std::fmt::Debug + 'static> Source for MemorySource<F> {
    fn clone_into_box(&self) -> Box<dyn Source + Send + Sync> {
        Box::new((*self).clone())
    }

    fn collect(&self) -> Result<config::Map<String, config::Value>, ConfigError> {
        let contents = self.content.as_str();

        self.format
            .parse(None, contents)
            .map_err(|cause| ConfigError::FileParse { uri: None, cause })
    }
}

impl Settings {
    pub fn new(config_file: Option<String>) -> Result<Self, ConfigError> {
        let config_builder = Config::builder()
            .add_source(MemorySource::new())
            .add_source(config::File::with_name("aaml"))
            .add_source(config::Environment::with_prefix("APP"));

        if let Some(config_file) = config_file {
            config_builder
                .add_source(config::File::with_name(&config_file))
                .build()?
                .try_deserialize::<Settings>()
        } else {
            config_builder.build()?.try_deserialize::<Settings>()
        }
    }
}
