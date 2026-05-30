use burn::data::network::downloader;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::PathBuf;

pub fn download() -> Result<PathBuf, io::Error> {
    const URL: &str = "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth";

    let model_dir = dirs::home_dir()
        .expect("Should be able to get a home directory")
        .join("cache")
        .join("sixdrepnet360-burn");

    if !model_dir.exists() {
        std::fs::create_dir_all(&model_dir)?;
    }

    let file_base_name = URL.rsplit_once('/').unwrap().1;
    let file_name = model_dir.join(file_base_name);
    if !file_name.exists() {
        // Download file content
        let bytes = downloader::download_file_as_bytes(URL, file_base_name);

        // Write content to file
        let mut output_file = File::create(&file_name)?;
        let bytes_written = output_file.write(&bytes)?;

        if bytes_written != bytes.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to write the whole model weights file.",
            ));
        }
    }

    Ok(file_name)
}
