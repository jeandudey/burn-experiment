use burn::backend::rocm::RocmDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Rocm, Wgpu};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{Mapper, MapperDataset, RngSource, ShuffledDataset};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use burn_3ddfa::{Pose, PoseDataset, PoseDatasetItem};
use eyre::{Result, WrapErr};
use mobilenetv2_burn::model::imagenet::Normalizer;
use resnet_burn::ResNet;
use resnet_burn::weights::ResNet50;

type TrainingDataset<D> = MapperDataset<D, ToTrainingDataItem, PoseDatasetItem>;

fn to_training_dataset<D>(dataset: D) -> TrainingDataset<D> {
    MapperDataset::new(dataset, ToTrainingDataItem)
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingDataItem {
    /// The cropped, resized head image in CHW order, ready for GPU upload.
    pub image_chw: Vec<f32>,
    /// The head pose.
    pub pose: Pose,
}

#[derive(Debug)]
pub struct ToTrainingDataItem;

impl Mapper<PoseDatasetItem, TrainingDataItem> for ToTrainingDataItem {
    fn map(&self, item: &PoseDatasetItem) -> TrainingDataItem {
        const K: f64 = 0.2;

        let PoseDatasetItem { image, annotations } = item;

        let pose = annotations
            .iter()
            .find_map(|v| v.as_pose())
            .expect("item doesn't have a pose")
            .clone();

        let landmarks_2d = annotations
            .iter()
            .find(|v| v.as_landmarks_2d().is_some())
            .map(|v| v.as_landmarks_2d())
            .flatten()
            .expect("item doens't have 2D landmarks");

        let [x0, y0, x1, y1] = landmarks_2d.to_bounding_box(K);
        let x0 = x0 as u32;
        let y0 = y0 as u32;
        let x1 = (x1 as u32).min(image.width());
        let y1 = (y1 as u32).min(image.height());

        let rgb32f = image.clone().into_rgb32f();
        let cropped = image::imageops::crop_imm(&rgb32f, x0, y0, x1 - x0, y1 - y0).to_image();
        let resized =
            image::imageops::resize(&cropped, 256, 256, image::imageops::FilterType::Nearest);
        let center = image::imageops::crop_imm(&resized, 16, 16, 224, 224).to_image();

        // Convert HWC → CHW so the batcher can upload the whole batch in one from_data call.
        let hwc = center.into_raw();
        let mut image_chw = Vec::with_capacity(3 * 224 * 224);
        for c in 0..3usize {
            for i in 0..(224 * 224) {
                image_chw.push(hwc[i * 3 + c]);
            }
        }

        TrainingDataItem { image_chw, pose }
    }
}

// ImageNet channel statistics — required by the pretrained backbone.
//const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
//const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub struct PoseBatch<B: Backend> {
    /// The images `[N, 3, 224, 224]`.
    pub images: Tensor<B, 4>,
    /// The targets `[N, 3]`.
    pub targets: Tensor<B, 2>,
}

pub struct PoseBatcher<B: Backend> {
    normalizer: Normalizer<B>,
}

impl<B: Backend> PoseBatcher<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            normalizer: Normalizer::<B>::new(device),
        }
    }
}

//fn rgba_to_chw_f32(img: &RgbaImage) -> Vec<f32> {
//    let (w, h) = (img.width() as usize, img.height() as usize);
//    debug_assert_eq!((w, h), (224, 224));
//    let mut out = vec![0f32; 3 * h * w];
//    for (i, px) in img.pixels().enumerate() {
//        let y = i / w;
//        let x = i % w;
//        // CHW + normalize to [0,1] + ImageNet mean/std
//        for c in 0..3 {
//            let v = px.0[c] as f32 / 255.0;
//            out[c * h * w + y * w + x] = (v - MEAN[c]) / STD[c];
//        }
//    }
//    out
//}

impl<B: Backend> Batcher<B, TrainingDataItem, PoseBatch<B>> for PoseBatcher<B> {
    fn batch(&self, items: Vec<TrainingDataItem>, device: &B::Device) -> PoseBatch<B> {
        let n = items.len();
        let mut pixels = Vec::with_capacity(n * 3 * 224 * 224);
        let mut targets = Vec::with_capacity(n * 3);

        for it in items {
            pixels.extend_from_slice(&it.image_chw);
            targets.extend_from_slice(&[
                it.pose.yaw as f32,
                it.pose.pitch as f32,
                it.pose.roll as f32,
            ]);
        }

        PoseBatch {
            images: self.normalizer.normalize(Tensor::<B, 4>::from_data(
                TensorData::new(pixels, [n, 3, 224, 224]),
                device,
            )),
            targets: Tensor::<B, 2>::from_data(TensorData::new(targets, [n, 3]), device),
        }
    }
}

#[derive(Module, Debug)]
pub struct PoseModel<B: Backend> {
    backbone: ResNet<B>,
}

impl<B: Backend> PoseModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let backbone = ResNet::<B>::resnet50_pretrained(ResNet50::ImageNet1kV2, device)
            .expect("Failed to download/load pretrained weights")
            .with_classes(3);
        Self { backbone }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        self.backbone.forward(x)
    }

    fn forward_step(&self, batch: PoseBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.images);
        let loss = MseLoss::new().forward(output.clone(), batch.targets.clone(), Reduction::Mean);
        RegressionOutput::new(loss, output, batch.targets)
    }
}

impl<B: AutodiffBackend> TrainStep for PoseModel<B> {
    type Input = PoseBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PoseBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        //let t = std::time::Instant::now();
        let item = self.forward_step(batch);
        let grads = item.loss.backward();
        //eprintln!("step: {:?}", t.elapsed());
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for PoseModel<B> {
    type Input = PoseBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PoseBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}

#[derive(Debug, Config)]
pub struct TrainingConfig {
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 32)] // 32
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)] // 1e-4
    pub lr: f64,
    pub optimizer: AdamConfig,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    device: B::Device,
    train_ds: impl Dataset<TrainingDataItem> + 'static,
    valid_ds: impl Dataset<TrainingDataItem> + 'static,
) -> Result<()> {
    let config = TrainingConfig::new(AdamConfig::new());
    B::seed(&device, config.seed);

    let batcher_train = PoseBatcher::<B>::new(&device);
    let batcher_valid = PoseBatcher::<B::InnerBackend>::new(&device);

    let dl_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_ds);

    let dl_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_ds);

    let result = SupervisedTraining::new(artifact_dir, dl_train, dl_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(Learner::new(
            PoseModel::new(&device),
            config.optimizer.init(),
            config.lr,
        ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .wrap_err("failed to save trained model")?;
    Ok(())
}

fn main() -> Result<()> {
    let train_dataset =
        PoseDataset::download_300w_lp().wrap_err("failed to download 300W-LP dataset")?;
    let train_dataset = to_training_dataset(train_dataset);
    let train_dataset = ShuffledDataset::new(train_dataset, RngSource::Default);

    let valid_dataset =
        PoseDataset::download_aflw2000_3d().wrap_err("failed to download AFLW2000-3D dataset")?;
    let valid_dataset = to_training_dataset(valid_dataset);
    let valid_dataset = ShuffledDataset::new(valid_dataset, RngSource::Default);

    if true {
        type RocmBackend = Autodiff<Rocm>;
        let device = RocmDevice::default();
        train::<RocmBackend>("./artifacts", device, train_dataset, valid_dataset)?;
    } else {
        type WgpuBackend = Autodiff<Wgpu>;
        let device = WgpuDevice::default();
        train::<WgpuBackend>("./artifacts", device, train_dataset, valid_dataset)?;
    }
    Ok(())
}
