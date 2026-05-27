use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::data::dataset::{Dataset, InMemDataset};
use image::{DynamicImage, ImageError};
use matio::Value;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::Level;
use tracing::event;
use tracing::span;
use walkdir::WalkDir;

/// Pose parameters, all values in radians.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    /// Pitch.
    pub pitch: f64,
    /// Yaw.
    pub yaw: f64,
    /// Roll.
    pub roll: f64,
    /// Translation `[tx, ty, tz]`.
    pub translation: [f64; 3],
    /// Scale.
    pub scale: f64,
}

impl Pose {
    pub fn from_var(var: &matio::Var) -> Result<Self, Error> {
        debug_assert!(var.name().unwrap().unwrap() == "Pose_Para");

        match var.value()? {
            Value::Single(v) => Self::from_single(v).ok_or(Error::InvalidVariableLength),
            Value::Double(v) => Self::from_double(v).ok_or(Error::InvalidVariableLength),
            _ => Err(Error::InvalidVariableType),
        }
    }

    pub fn from_double(v: &[f64]) -> Option<Self> {
        if v.len() < 7 {
            return None;
        }

        Some(Self {
            pitch: v[0],
            yaw: v[1],
            roll: v[2],
            translation: [v[3], v[4], v[5]],
            scale: v[6],
        })
    }

    pub fn from_single(v: &[f32]) -> Option<Self> {
        if v.len() < 7 {
            return None;
        }

        Some(Self {
            pitch: f64::from(v[0]),
            yaw: f64::from(v[1]),
            roll: f64::from(v[2]),
            translation: [f64::from(v[3]), f64::from(v[4]), f64::from(v[5])],
            scale: f64::from(v[6]),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    Pose(Pose),
    Landmarks2d([[f64; 68]; 2]),
}

impl Annotation {
    pub fn from_var(var: &matio::Var) -> Result<Option<Self>, Error> {
        let name = match var.name() {
            Ok(Some(v)) => v,
            Ok(None) => return Ok(None),
            Err(e) => {
                event!(Level::ERROR, "mat variable name contains interior NUL: {e}");
                return Ok(None);
            }
        };

        match name {
            "Pose_Para" => Pose::from_var(var).map(Annotation::Pose).map(Some),
            "pt2d" => match var.value()? {
                Value::Double(v) => {
                    if v.len() < 136 {
                        return Err(Error::InvalidVariableLength);
                    }

                    let mut landmarks = [[0.0; 68]; 2];
                    landmarks[0].copy_from_slice(&v[0..68]);
                    landmarks[1].copy_from_slice(&v[68..136]);
                    Ok(Some(Annotation::Landmarks2d(landmarks)))
                }
                _ => Err(Error::InvalidVariableType),
            },
            _ => Ok(None),
        }
    }

    pub fn as_pose(&self) -> Option<&Pose> {
        match self {
            Annotation::Pose(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_landmarks_2d(&self) -> Option<&[[f64; 68]; 2]> {
        match self {
            Annotation::Landmarks2d(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseDatasetItem {
    pub image: DynamicImage,
    pub annotations: Vec<Annotation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseDatasetItemRaw {
    pub image_path: PathBuf,
    pub mat_path: PathBuf,
}

impl PoseDatasetItemRaw {
    pub fn load(&self) -> Result<PoseDatasetItem, Error> {
        let mut mat = matio::open(&self.mat_path, matio::Access::Read)?;
        let annotations = mat
            .vars()
            .filter_map(|var| Annotation::from_var(&var).transpose())
            .collect::<Result<Vec<_>, _>>()?;

        let image = image::open(&self.image_path)?;

        Ok(PoseDatasetItem { image, annotations })
    }
}

fn find_images(images_path: impl AsRef<Path>) -> Result<Vec<PoseDatasetItemRaw>, Error> {
    let span = span!(Level::TRACE, "find_images");
    let _enter = span.enter();

    let mut items = Vec::new();

    for entry in WalkDir::new(images_path) {
        let entry = entry?;
        let entry = entry.path();
        if entry.extension().map(|e| e == "jpg").unwrap_or(false) {
            let mat_path = entry.with_extension("mat");
            if mat_path.exists() {
                items.push(PoseDatasetItemRaw {
                    image_path: entry.to_path_buf(),
                    mat_path,
                });
            } else {
                event!(
                    parent: &span,
                    Level::WARN,
                    entry = entry.display().to_string(),
                    "jpg file does not have a mat file",
                );
            }
        }
    }

    event!(parent: &span, Level::TRACE, "finished reading dataset directory");
    Ok(items)
}

struct PathToPoseDatasetItem;

impl Mapper<PoseDatasetItemRaw, PoseDatasetItem> for PathToPoseDatasetItem {
    fn map(&self, item_raw: &PoseDatasetItemRaw) -> PoseDatasetItem {
        match item_raw.load() {
            Ok(item) => item,
            Err(e) => {
                panic!("failed to load item: {e}");
            }
        }
    }
}

type PoseDatasetMapper =
    MapperDataset<InMemDataset<PoseDatasetItemRaw>, PathToPoseDatasetItem, PoseDatasetItemRaw>;

pub struct PoseDataset {
    dataset: PoseDatasetMapper,
}

impl Dataset<PoseDatasetItem> for PoseDataset {
    fn get(&self, index: usize) -> Option<PoseDatasetItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl PoseDataset {
    pub fn new(images_path: impl AsRef<Path>) -> Result<Self, Error> {
        let items = find_images(images_path)?;
        let in_mem_dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(in_mem_dataset, PathToPoseDatasetItem);
        Ok(Self { dataset })
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error")]
    Io(#[from] io::Error),
    #[error("failed to walk directory")]
    WalkDir(#[from] walkdir::Error),
    #[error("failed to load mat file")]
    Matio(#[from] matio::Error),
    #[error("failed to load image")]
    Image(#[from] ImageError),
    #[error("invalid variable type in mat file")]
    InvalidVariableType,
    #[error("invalid variable length in mat file")]
    InvalidVariableLength,
}
