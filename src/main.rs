use burn::backend::rocm::RocmDevice;
use burn::backend::{Autodiff, Rocm};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{
    Mapper, MapperDataset, PartialDataset, RngSource, ShuffledDataset,
};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use burn_300w_lp::{Pose, PoseDataset, PoseDatasetItem};
use image::imageops::{FilterType, resize};
use image::{GenericImageView, RgbaImage};
use resnet_burn::ResNet;
use resnet_burn::weights::ResNet50;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct CroppedPoseDatasetItem {
    pub image: RgbaImage,
    pub pose: Pose,
}

#[derive(Debug)]
pub struct CenterCrop;

impl Mapper<PoseDatasetItem, CroppedPoseDatasetItem> for CenterCrop {
    fn map(&self, item: &PoseDatasetItem) -> CroppedPoseDatasetItem {
        const K: f64 = 0.2;

        let PoseDatasetItem { image, annotations } = item;

        let landmarks_2d = annotations
            .iter()
            .find(|v| v.as_landmarks_2d().is_some())
            .map(|v| v.as_landmarks_2d())
            .flatten()
            .expect("item doens't have 2D landmarks");

        let x_min = landmarks_2d[0]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let y_min = landmarks_2d[1]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let x_max = landmarks_2d[0].iter().copied().reduce(f64::max).unwrap();
        let y_max = landmarks_2d[0].iter().copied().reduce(f64::max).unwrap();

        let x0 = (x_min - (2.0 * K * (x_max - x_min).abs())) as u32;
        let y0 = (y_min - (2.0 * K * (y_max - y_min).abs())) as u32;
        let x1 = (x_max + (2.0 * K * (x_max - x_min).abs())) as u32;
        let y1 = (y_min + (0.6 * K * (y_max - y_min).abs())) as u32;

        let x1 = x1.min(image.width());
        let y1 = y1.min(image.width());

        let sub_image = image.view(x0, y0, x1 - x0, y1 - y0).to_image();
        let resized = resize(&sub_image, 256, 256, FilterType::Nearest);
        let image = resized.view(16, 16, 224, 224).to_image();

        let pose = annotations
            .iter()
            .find(|v| v.as_pose().is_some())
            .map(|v| v.as_pose())
            .flatten()
            .unwrap()
            .clone();

        CroppedPoseDatasetItem { image, pose }
    }
}

/// ImageNet channel statistics — required by the pretrained backbone.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub struct PoseBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2>,
}

pub struct PoseBatcher;

fn rgba_to_chw_f32(img: &RgbaImage) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    debug_assert_eq!((w, h), (224, 224));
    let mut out = vec![0f32; 3 * h * w];
    for (i, px) in img.pixels().enumerate() {
        let y = i / w;
        let x = i % w;
        // CHW + normalize to [0,1] + ImageNet mean/std
        for c in 0..3 {
            let v = px.0[c] as f32 / 255.0;
            out[c * h * w + y * w + x] = (v - MEAN[c]) / STD[c];
        }
    }
    out
}

impl<B: Backend> Batcher<B, CroppedPoseDatasetItem, PoseBatch<B>> for PoseBatcher {
    fn batch(&self, items: Vec<CroppedPoseDatasetItem>, device: &B::Device) -> PoseBatch<B> {
        let n = items.len();

        let mut images = Vec::with_capacity(n * 3 * 224 * 224);
        let mut targets = Vec::with_capacity(n * 3);

        for it in &items {
            images.extend(rgba_to_chw_f32(&it.image));
            targets.extend_from_slice(&[
                it.pose.yaw as f32,
                it.pose.pitch as f32,
                it.pose.roll as f32,
            ])
        }

        PoseBatch {
            images: Tensor::<B, 4>::from_data(
                TensorData::new(images, [n, 3, 224, 224]).convert::<B::FloatElem>(),
                device,
            ),
            targets: Tensor::<B, 2>::from_data(
                TensorData::new(targets, [n, 3]).convert::<B::FloatElem>(),
                device,
            ),
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
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
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
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub optimizer: AdamConfig,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    device: B::Device,
    train_ds: impl Dataset<CroppedPoseDatasetItem> + 'static,
    valid_ds: impl Dataset<CroppedPoseDatasetItem> + 'static,
) {
    let config = TrainingConfig::new(AdamConfig::new());
    B::seed(&device, config.seed);

    let dl_train = DataLoaderBuilder::<B, _, _>::new(PoseBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_ds);

    let dl_valid = DataLoaderBuilder::new(PoseBatcher)
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
        .expect("failed to save trained model");
}

fn main() {
    let dataset = MapperDataset::new(PoseDataset::new("datasets/300W_LP").unwrap(), CenterCrop);
    let shuffled_dataset = Arc::new(ShuffledDataset::new(dataset, RngSource::Default));
    let cut = shuffled_dataset.len() * 9 / 10;
    let train_dataset = PartialDataset::new(shuffled_dataset.clone(), 0, cut);
    let valid_dataset = PartialDataset::new(shuffled_dataset.clone(), cut, shuffled_dataset.len());

    type RocmBackend = Autodiff<Rocm>;
    let device = RocmDevice::default();
    train::<RocmBackend>("./artifacts", device, train_dataset, valid_dataset);
}
