use crate::downloader::{download_300w_lp_as_file, extract_zip};
use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::data::dataset::{Dataset, InMemDataset};
use burn::data::network::downloader::download_file_as_bytes;
use image::{DynamicImage, ImageError};
use matio::{Value, Var};
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{Level, event, span};
use walkdir::WalkDir;
use zip::result::ZipError;

mod downloader;

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
    pub fn from_var(var: &Var) -> Result<Self, Error> {
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

/// 68-point 2D landmarks.
#[derive(Debug, Clone, PartialEq)]
pub struct Landmarks2d(pub [[f64; 68]; 2]);

impl Landmarks2d {
    /// Parse the 2D landmarks from a MAT file variable.
    pub fn from_var(var: &Var) -> Result<Self, Error> {
        let value = var.value()?;
        let v = value.as_double().ok_or(Error::InvalidVariableType)?;
        Self::from_slice(v)
    }

    /// Parse the 2D landmarks from a slice.
    ///
    /// # Notes
    ///
    /// The slice must have at least 136 elements (`68 * 2`) where
    /// the first 68 elements are for the X axis and 68 for the Y axis.
    pub fn from_slice(v: &[f64]) -> Result<Self, Error> {
        if v.len() < 136 {
            return Err(Error::InvalidVariableLength);
        }

        let mut landmarks = [[0.0; 68]; 2];
        landmarks[0].copy_from_slice(&v[0..68]);
        landmarks[1].copy_from_slice(&v[68..136]);
        Ok(Self(landmarks))
    }

    /// Compute the bounding box from the 2D landmarks.
    ///
    /// The `margin` is the fraction by which each side gets expanded relative
    /// to the box.
    ///
    /// # Return
    ///
    /// The return value is `[x_min, y_min, x_max, y_max]`.
    ///
    /// # Notes
    ///
    /// The values can be outside of the image size.
    pub fn to_bounding_box(&self, margin: f64) -> [f64; 4] {
        const SIDE: f64 = 0.6;
        const TOP: f64 = 2.0;

        let xs = &self.0[0];
        let ys = &self.0[1];

        let x_min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let w = x_max - x_min;
        let h = y_max - y_min;

        [
            x_min - SIDE * margin * w,
            y_min - TOP * margin * h,
            x_max + SIDE * margin * w,
            y_max + SIDE * margin * h,
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    Pose(Pose),
    Landmarks2d(Landmarks2d),
}

impl Annotation {
    pub fn from_var(var: &Var) -> Result<Option<Self>, Error> {
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
            "pt2d" => Landmarks2d::from_var(var)
                .map(Annotation::Landmarks2d)
                .map(Some),
            _ => Ok(None),
        }
    }

    /// Extract the pose or `None` if it isn't a pose.
    pub fn as_pose(&self) -> Option<&Pose> {
        match self {
            Annotation::Pose(v) => Some(v),
            _ => None,
        }
    }

    /// Extract the 2D landmarks or `None` if it isn't 2D landmarks.
    pub fn as_landmarks_2d(&self) -> Option<&Landmarks2d> {
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
    /// Creates a new pose dataset from the given images path to the
    /// dataset.
    pub fn new(images_path: impl AsRef<Path>) -> Result<Self, Error> {
        let items = find_images(images_path)?;
        let in_mem_dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(in_mem_dataset, PathToPoseDatasetItem);
        Ok(Self { dataset })
    }

    /// Downloads the 300W-LP dataset.
    pub fn download_300w_lp() -> Result<Self, Error> {
        let dataset_dir = Self::cache_dir()?;

        let file_name = dataset_dir.join("300W-LP.zip");
        if !file_name.exists() {
            download_300w_lp_as_file(&file_name)?;
        }

        let root_dir = dataset_dir.join("300W_LP");
        if !root_dir.exists() {
            extract_zip(&file_name, &dataset_dir)?;
        }

        Self::new(root_dir)
    }

    /// Downloads the AFLW2000-3D dataset.
    pub fn download_aflw2000_3d() -> Result<Self, Error> {
        const URL: &str =
            "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip";

        let dataset_dir = Self::cache_dir()?;

        let file_base_name = URL.rsplit_once('/').unwrap().1;
        let file_name = dataset_dir.join(file_base_name);
        if !file_name.exists() {
            let bytes = download_file_as_bytes(URL, file_base_name);

            let mut output_file = File::create(&file_name)?;
            output_file.write_all(&bytes)?;
        }

        let root_dir = dataset_dir.join("AFLW2000");
        if !root_dir.exists() {
            extract_zip(file_name, &dataset_dir)?;
        }

        Self::new(root_dir)
    }

    fn cache_dir() -> Result<PathBuf, Error> {
        let dataset_dir = dirs::home_dir()
            .ok_or(Error::NoHomeDirectory)?
            .join(".cache")
            .join("burn-3ddfa");

        if !dataset_dir.exists() {
            std::fs::create_dir_all(&dataset_dir)?;
        }

        Ok(dataset_dir)
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error")]
    Io(#[from] io::Error),
    #[error("failed to walk directory")]
    WalkDir(#[from] walkdir::Error),
    #[error("failed to send request")]
    Reqwest(#[from] reqwest::Error),
    #[error("failed to load mat file")]
    Matio(#[from] matio::Error),
    #[error("failed to load image")]
    Image(#[from] ImageError),
    #[error("failed extract zip archive")]
    Zip(#[from] ZipError),
    #[error("invalid variable type in mat file")]
    InvalidVariableType,
    #[error("invalid variable length in mat file")]
    InvalidVariableLength,
    #[error("no home directory found to download the dataset file")]
    NoHomeDirectory,
}
