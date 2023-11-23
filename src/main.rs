use std::fs;
use clap::Parser;
use aaml::{ config::Settings, cli::Cli };

#[tokio::main]
async fn main() {

    let cli = Cli::parse();
    let mut settings = Settings::new(cli.clone().config_file).unwrap();

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
