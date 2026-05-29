use crate::Error;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use std::io;
use std::path::Path;
use tokio::io::AsyncWriteExt;

#[tokio::main(flavor = "current_thread")]
pub async fn download_300w_lp_as_file(file_name: impl AsRef<Path>) -> Result<(), Error> {
    const URL: &str = "https://drive.google.com/uc?export=download&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k";

    let client = reqwest::Client::builder().cookie_store(true).build()?;
    let resp = client.get(URL).send().await?;
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .to_string();

    let mut response = if ct.starts_with("text/html") {
        let html = resp.text().await?;
        let (action, params) = parse_download_form(&html);
        client.get(action).query(&params).send().await?
    } else {
        resp
    };

    let mut file = tokio::fs::File::create(&file_name).await?;

    // TODO: If not available maybe provide another progress indicator.
    let total_size = response
        .content_length()
        .expect("content length is not available");

    // Pretty progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::with_template(
            "300-LW.zip\n    {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .with_key(
            "eta",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            },
        )
        .progress_chars("▬  "),
    );

    let mut downloaded = 0;
    while let Some(chunk) = response.chunk().await? {
        let new = total_size.min(downloaded + chunk.len() as u64);
        file.write_all(&chunk).await?;
        downloaded = new;
        pb.set_position(new);
    }

    file.flush().await?;

    Ok(())
}

fn parse_download_form(html: &str) -> (String, Vec<(String, String)>) {
    use scraper::{Html, Selector};
    let doc = Html::parse_document(html);

    let form_sel = Selector::parse("form#download-form").unwrap();
    let form = doc.select(&form_sel).next().unwrap();
    let action = form
        .value()
        .attr("action")
        .expect("form has no action")
        .to_string();

    let input_sel = Selector::parse("input[name]").unwrap();
    let params = form
        .select(&input_sel)
        .filter_map(|i| {
            let name = i.value().attr("name")?;
            let value = i.value().attr("value").unwrap_or("");
            Some((name.to_string(), value.to_string()))
        })
        .collect();

    (action, params)
}

/// TODO: Probably could use ripunzip to improve speed, but it conflicts with
/// tracel-llvm-bundler (liblzma).
///
/// Update dependencies of ripunzip to fix this.
pub fn extract_zip(file_name: impl AsRef<Path>, output_dir: impl AsRef<Path>) -> Result<(), Error> {
    let file = std::fs::File::open(&file_name)?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Sum total uncompressed size across all entries
    let mut total_bytes: u64 = 0;
    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index(i) {
            total_bytes += entry.size();
        }
    }

    let pb = ProgressBar::new(total_bytes);
    pb.set_style(
        ProgressStyle::with_template(
            "extracting {prefix}\n    {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .with_key(
            "eta",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            },
        )
        .progress_chars("▬  "),
    );

    if let Some(file_name) = file_name.as_ref().file_name() {
        pb.set_prefix(file_name.to_string_lossy().to_string());
    }

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let out_path = output_dir.as_ref().join(entry.enclosed_name().unwrap());

        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut out_file = std::fs::File::create(&out_path)?;
            io::copy(&mut entry, &mut pb.wrap_write(&mut out_file))?;
        }
    }

    Ok(())
}
