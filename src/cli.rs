use std::{ path::PathBuf, fs };
use clap::{ Parser, ValueEnum };

use crate::{
    config::Settings,
    aaml::AutoAnnotation,
    model::coco::{ CoCoAnnotations, Image, Category },
};

#[derive(Parser)]
#[command(name = "aaml-cli")]
#[command(version = "1.0")]
#[command(about = "Auto annotation ml cli", long_about = None)]
pub struct Cli {
    /// 数据集
    pub dataset: PathBuf,

    #[arg(long)]
    #[arg(value_enum)]
    #[arg(default_value_t = Device::Cpu)]
    pub device: Device,
    /// yolov8 模型文件,只支持 safetensors 格式
    #[arg(long)]
    pub model_file: Option<String>,
    /// 标签
    #[arg(long)]
    pub labels: Option<Vec<String>>,

    /// 配置文件
    #[arg(long)]
    pub config_file: Option<String>,
}

impl Cli {
    pub async fn run(&self, settings: Settings) -> anyhow::Result<CoCoAnnotations> {
        if !self.dataset.is_dir() {
            return Err(anyhow::anyhow!("不是有效的文件夹"));
        }
        let mut coco = CoCoAnnotations { ..Default::default() };
        coco.set_categories(
            settings.labels
                .iter()
                .enumerate()
                .map(|(index, v)| Category {
                    id: (index + 1) as i64,
                    name: v.to_string(),
                    supercategory: "".to_string(),
                })
                .collect()
        );
        let aaml = AutoAnnotation::new(settings)?;
        let anid = 0;
        for (index, entry) in fs::read_dir(self.dataset.as_path())?.enumerate() {
            let path = entry?.path();
            // dbg!(&path);
            if path.is_file() && path.extension() == Some("jpg".as_ref()) {
                let image = image::open(path.as_path())?;
                let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                let image_id = index + 1;
                coco.add_image(Image {
                    id: image_id as i64,
                    width: image.width() as i64,
                    height: image.height() as i64,
                    file_name: filename,
                    ..Default::default()
                });
                let predictions = aaml.predict(&image)?;
                let annotations = aaml.post_process(anid, image_id as u64, predictions);
                coco.add_annotations(annotations);
            }
        }

        Ok(coco)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum Device {
    Cpu,
    Cuda,
}
