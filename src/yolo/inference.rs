use std::path::Path;
use candle_core::{ Tensor, Device, DType, Module, IndexOp };
use candle_nn::VarBuilder;
use crate::yolo::v8::Multiples;

use super::{object_detection::{ Bbox, KeyPoint, non_maximum_suppression }, v8::YoloV8};
use image::DynamicImage;


#[derive(Clone, Copy, Debug)]
pub enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Clone, Debug)]
pub struct Inference {
    model: YoloV8,
}


impl Inference {
    pub fn load<S: AsRef<Path>>(
        model_file: S,
        weight_size: Which,
        num_classes: usize
    ) -> candle_core::Result<Self> {
        let multiples = match weight_size {
            Which::N => Multiples::n(),
            Which::S => Multiples::s(),
            Which::M => Multiples::m(),
            Which::L => Multiples::l(),
            Which::X => Multiples::x(),
        };
        let device = if cfg!(feature = "cuda") { Device::new_cuda(0)? } else { Device::Cpu };

        // 检查 cuda 是否工作
        #[cfg(feature = "cuda")]
        info!("cuda 状态: {}", candle_core::utils::cuda_is_available());

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = YoloV8::load(vb, multiples, num_classes)?;
        Ok(Self {
            model,
        })
    }

    pub fn predict(
        &self,
        original_image: &DynamicImage
    ) -> candle_core::Result<Vec<Vec<Bbox<Vec<KeyPoint>>>>> {
        // let original_image = image::io::Reader::open(&image_name)?
        //     .decode()
        //     .map_err(candle_core::Error::wrap)?;
        let device = if cfg!(feature = "cuda") { Device::new_cuda(0)? } else { Device::Cpu };
        
        let confidence_threshold = 0.25f32;
        let nms_threshold = 0.45f32;
        let image_t = {
            let (width, height) = {
                let w = original_image.width() as usize;
                let h = original_image.height() as usize;
                if w < h {
                    let w = (w * 640) / h;
                    // Sizes have to be divisible by 32.
                    ((w / 32) * 32, 640)
                } else {
                    let h = (h * 640) / w;
                    (640, (h / 32) * 32)
                }
            };
            

            // intel cpu 下
            // 这个操作使用 三次样条插值（INTER_CUBIC）-> FilterType::CatmullRom 时非常的耗时,大约在100ms -120ms
            //
            // 使用最临近插值（INTER_NEAREST）-> FilterType::Nearest 40ms-80ms
            // 双线性插值（INTER_LINEAR）-> FilterType::Triangle 70ms-100ms
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::Nearest
            );

            let data = img.to_rgb8().into_raw();
            
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &device
            )?.permute((2, 0, 1))?
        };

        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1.0 / 255.0))?;
        let predictions = self.model.forward(&image_t)?.squeeze(0)?;

        //后处理
        let (pred_size, npreds) = predictions.dims2()?;
        let nclasses = pred_size - 4;

        //按（最大）类索引分组的边界框
        let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
        //提取置信度高于阈值的边界框

        for index in 0..npreds {
            let p = predictions.i((.., index))?;

            // let sstart_time = Instant::now();
            // 这行代码有性能问题，一次耗时 30微秒
            let pred = Vec::<f32>::try_from(p)?;
            let confidence = *pred
                .iter()
                .skip(4)
                .max_by(|x, y| x.total_cmp(y))
                .unwrap();
            // let pred = Vec::<f32>::try_from(p)?;
            // let confidence = *pred.iter().skip(4).max_by(|x, y| x.total_cmp(y)).unwrap();
            // if cfg!(feature = "log") {
            //     info!("循环次数{},耗时: {}微秒", index,sstart_time.elapsed().as_micros());
            // }

            if confidence > confidence_threshold {
                let mut class_index = 0;
                for i in 0..nclasses {
                    if pred[4 + i] > pred[4 + class_index] {
                        class_index = i;
                    }
                }
                if pred[class_index + 4] > 0.0 {
                    let bbox = Bbox {
                        xmin: pred[0] - pred[2] / 2.0,
                        ymin: pred[1] - pred[3] / 2.0,
                        xmax: pred[0] + pred[2] / 2.0,
                        ymax: pred[1] + pred[3] / 2.0,
                        confidence,
                        data: vec![],
                    };
                    bboxes[class_index].push(bbox);
                }
            }
        }
        non_maximum_suppression(&mut bboxes, nms_threshold);
        
        Ok(bboxes)
    }

}
