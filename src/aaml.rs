use candle_core::Result;
use image::{ DynamicImage, imageops };

use crate::{
    yolo::{ inference::{ Inference, Which }, object_detection::{ Bbox, KeyPoint } },
    config::Settings,
    model::coco::{ CoCoAnnotations, Category, Image, Annotation, self },
};

#[derive(Debug)]
pub struct AutoAnnotation {
    config: Settings,
    infer: Inference,
}

impl AutoAnnotation {
    pub fn new(config: Settings) -> Result<Self> {
        let infer = Inference::load(&config.model_path, Which::M, config.labels.len())?;

        Ok(Self { config, infer })
    }

    pub fn predict(&self, original_image: &DynamicImage) -> Result<Vec<Vec<Bbox<Vec<KeyPoint>>>>> {
        self.infer.predict(original_image)
    }

    pub fn post_process(
        &self,
        mut anid: u64,
        image_id: u64,
        predictions: Vec<Vec<Bbox<Vec<KeyPoint>>>>
    ) -> Vec<Annotation> {
        //default coco
        // let mut coco = CoCoAnnotations {..Default::default() };
        let mut annotations = vec![];

        for (index, bboxs) in predictions.iter().enumerate() {
            let label_index = index + 1;

            for bbox in bboxs {
                let annotation = Annotation {
                    id: anid,
                    image_id,
                    category_id: label_index as i64,
                    bbox: vec![bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin], //[x,y,w,h]
                    ..Default::default()
                };
                annotations.push(annotation);
                anid += 1;
            }
        }
        annotations
    }
}
