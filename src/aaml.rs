use candle_core::Result;
use image::DynamicImage;

use crate::{
    yolo::{ inference::Inference, object_detection::{ Bbox, KeyPoint } },
    config::Settings,
    model::coco::Annotation,
};

#[derive(Debug)]
pub struct AutoAnnotation {
    infer: Inference,
}

impl AutoAnnotation {
    pub fn new(config: Settings) -> Result<Self> {
        let confidence_threshold = config.yolo.confidence_threshold;
        let nms_threshold = config.yolo.nms_threshold;
        let infer = Inference::load(
            &config.yolo.model_path,
            config.yolo.weight_size.into(),
            confidence_threshold,
            nms_threshold,
            config.yolo.labels.len()
        )?;

        Ok(Self {infer })
    }

    pub fn predict(&self, original_image: &DynamicImage) -> Result<Vec<Vec<Bbox<Vec<KeyPoint>>>>> {
        self.infer.predict(original_image)
    }

    pub fn post_process(
        &self,
        mut anid: u64,
        image_id: u64,
        width: u64,
        height: u64,
        predictions: Vec<Vec<Bbox<Vec<KeyPoint>>>>
    ) -> Vec<Annotation> {
        //default coco
        // let mut coco = CoCoAnnotations {..Default::default() };
        let mut annotations = vec![];

        for (index, bboxs) in predictions.iter().enumerate() {
            let label_index = index + 1;

            let (initial_h, initial_w) = (height, width);
            let (width, height) = {
                let w = width as usize;
                let h = height as usize;
                if w < h {
                    let w = (w * 640) / h;
                    // Sizes have to be divisible by 32.
                    ((w / 32) * 32, 640)
                } else {
                    let h = (h * 640) / w;
                    (640, (h / 32) * 32)
                }
            };
            let w_ratio = (initial_w as f32) / (width as f32);
            let h_ratio = (initial_h as f32) / (height as f32);

            for bbox in bboxs {
                let annotation = Annotation {
                    id: anid,
                    image_id,
                    category_id: label_index as i64,
                    bbox: vec![
                        bbox.xmin * w_ratio,
                        bbox.ymin * h_ratio,
                        (bbox.xmax - bbox.xmin) * w_ratio,
                        (bbox.ymax - bbox.ymin) * h_ratio
                    ], //[x,y,w,h]
                    ..Default::default()
                };
                annotations.push(annotation);
                anid += 1;
            }
        }
        annotations
    }
}
