// Example code that deserializes and serializes the model.
// extern crate serde;
// #[macro_use]
// extern crate serde_derive;
// extern crate serde_json;
//
// use generated_module::Person;
//
// fn main() {
//     let json = r#"{"answer": 42}"#;
//     let model: Person = serde_json::from_str(&json).unwrap();
// }

use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize,Default)]
pub struct CoCoAnnotations {
    #[serde(rename = "licenses")]
    pub licenses: Vec<License>,

    #[serde(rename = "info")]
    pub info: Info,

    #[serde(rename = "categories")]
    pub categories: Vec<Category>,

    #[serde(rename = "images")]
    pub images: Vec<Image>,

    #[serde(rename = "annotations")]
    pub annotations: Vec<Annotation>,
}

impl CoCoAnnotations {
    pub fn set_images(&mut self, images: Vec<Image>) {
        self.images = images;
    }

    pub fn add_image(&mut self, image: Image) {
        self.images.push(image);
    }

    pub fn add_annotations(&mut self, annotation: Vec<Annotation>) {
        self.annotations.extend(annotation);
    }

    pub fn set_categories(&mut self, categories: Vec<Category>) {
        self.categories = categories;
    }
}

#[derive(Serialize, Deserialize,Default)]
pub struct Annotation {
    #[serde(rename = "id")]
    pub id: u64,

    #[serde(rename = "image_id")]
    pub image_id: u64,

    #[serde(rename = "category_id")]
    pub category_id: i64,

    #[serde(rename = "segmentation")]
    pub segmentation: Vec<Option<serde_json::Value>>,

    #[serde(rename = "area")]
    pub area: f64,

    #[serde(rename = "bbox")]
    pub bbox: Vec<f32>,

    #[serde(rename = "iscrowd")]
    pub iscrowd: i64,

    #[serde(rename = "attributes")]
    pub attributes: Attributes,
}

#[derive(Serialize, Deserialize,Default,Debug)]
pub struct Attributes {
    #[serde(rename = "occluded")]
    pub occluded: bool,

    #[serde(rename = "rotation")]
    pub rotation: i64,
}

#[derive(Serialize, Deserialize)]
pub struct Category {
    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "name")]
    pub name: String,

    #[serde(rename = "supercategory")]
    pub supercategory: String,
}

#[derive(Serialize, Deserialize,Debug,Default)]
pub struct Image {
    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "width")]
    pub width: i64,

    #[serde(rename = "height")]
    pub height: i64,

    #[serde(rename = "file_name")]
    pub file_name: String,

    #[serde(rename = "license")]
    pub license: i64,

    #[serde(rename = "flickr_url")]
    pub flickr_url: String,

    #[serde(rename = "coco_url")]
    pub coco_url: String,

    #[serde(rename = "date_captured")]
    pub date_captured: i64,
}

#[derive(Serialize, Deserialize,Default)]
pub struct Info {
    #[serde(rename = "contributor")]
    pub contributor: String,

    #[serde(rename = "date_created")]
    pub date_created: String,

    #[serde(rename = "description")]
    pub description: String,

    #[serde(rename = "url")]
    pub url: String,

    #[serde(rename = "version")]
    pub version: String,

    #[serde(rename = "year")]
    pub  year: String,
}

#[derive(Serialize, Deserialize,Default)]
pub struct License {
    #[serde(rename = "name")]
    pub name: String,

    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "url")]
    pub url: String,
}
