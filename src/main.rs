use std::fs;

use clap::Parser;
use config::Config;
use aaml::{ config::Settings, cli::Cli };

#[tokio::main]
async fn main() {
    let config_builder = Config::builder()
        .add_source(config::File::with_name("default"))
        .add_source(config::Environment::with_prefix("APP"));

    let cli = Cli::parse();
    let mut settings = if let Some(ref config_file) = cli.config_file {
        config_builder
            .add_source(config::File::with_name(config_file))
            .build()
            .unwrap()
            .try_deserialize::<Settings>()
            .unwrap()
    } else {
        config_builder.build().unwrap().try_deserialize::<Settings>().unwrap()
    };

    if let Some(labels) = &cli.labels {
        settings.yolo.labels = labels.to_vec();
    }

    let coco = cli.run(settings).await;
    match coco {
        Ok(data) => {
            
            let contents = serde_json::to_string(&data).unwrap();
            fs::write("instances_default.json", contents.as_bytes()).unwrap();
        }
        Err(e) => eprintln!("{:?}", e),
    }
}
