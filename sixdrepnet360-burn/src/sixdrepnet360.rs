use crate::block::{LayerBlock, LayerBlockConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
    pool::{AvgPool2d, AvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
};
use burn::prelude::*;

#[cfg(feature = "pretrained")]
use crate::weights;
#[cfg(feature = "pretrained")]
use burn_store::{ModuleSnapshot, PytorchStore, PytorchStoreError};

/// 6DRepNet360 model.
#[derive(Debug, Module)]
pub struct SixDRepNet360<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    maxpool: MaxPool2d,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    avgpool: AvgPool2d,
    linear_reg: Linear<B>,
}

impl<B: Backend> SixDRepNet360<B> {
    pub fn new(layers: [usize; 4], device: &B::Device) -> Self {
        SixDRepNet360Config::new(layers).init(device)
    }
}

#[cfg(feature = "pretrained")]
impl<B: Backend> SixDRepNet360<B> {
    /// Download a pretrained 6DRepNet360 model from a PyTorch weights file.
    #[cfg(feature = "pretrained")]
    pub fn pretrained(device: &B::Device) -> Result<Self, PytorchStoreError> {
        let mut model = Self::new([3, 4, 6, 3], device);
        Self::download_weights(&mut model)?;
        Ok(model)
    }

    /// Download the pretrained weights for the model.
    pub fn download_weights(model: &mut Self) -> Result<(), PytorchStoreError> {
        let torch_weights = weights::download().map_err(|err| {
            PytorchStoreError::Other(format!("Could not download weights.\nError: {err}"))
        })?;

        let mut store = PytorchStore::from_file(torch_weights)
            // Map *.downsample.0.* -> *.downsample.conv.*
            .with_key_remapping("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
            // Map *.downsample.1.* -> *.downsample.bn.*
            .with_key_remapping("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
            // Map layer[i].[j].* -> layer[i].blocks.[j].*
            .with_key_remapping("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");
        model.load_from(&mut store)?;
        Ok(())
    }
}

/// Configuration for the 6DRepNet360 model.
#[derive(Debug)]
pub struct SixDRepNet360Config {
    pub layers: [usize; 4],
}

impl SixDRepNet360Config {
    /// Create a new configuration with the given layer sizes.
    pub fn new(layers: [usize; 4]) -> Self {
        Self { layers }
    }

    /// Initialize the model with the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SixDRepNet360<B> {
        const EXPANSION: usize = 4;

        SixDRepNet360 {
            conv1: Conv2dConfig::new([3, 64], [7, 7])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
                .with_bias(false)
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            relu: Relu::new(),
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(),
            layer1: LayerBlockConfig::new(self.layers[0], 64, 64 * EXPANSION, 1, true).init(device),
            layer2: LayerBlockConfig::new(self.layers[1], 64 * EXPANSION, 128 * EXPANSION, 2, true)
                .init(device),
            layer3: LayerBlockConfig::new(
                self.layers[2],
                128 * EXPANSION,
                256 * EXPANSION,
                2,
                true,
            )
            .init(device),
            layer4: LayerBlockConfig::new(
                self.layers[3],
                256 * EXPANSION,
                512 * EXPANSION,
                2,
                true,
            )
            .init(device),
            avgpool: AvgPool2dConfig::new([7, 7]).init(),
            linear_reg: LinearConfig::new(512 * EXPANSION, 6).init(device),
        }
    }
}
