use std::{io::Read, path::PathBuf, sync::Arc};

/// This module manages downloading a dataset from Hugging Face. I'm using the [OpenWebText dataset](https://huggingface.co/datasets/Skylion007/openwebtext), and this is hard-coded for now.

use clap::{Arg, Command};
use futures_util::StreamExt;
use tokio::{io::AsyncWriteExt, task::JoinSet};

use crate::DATA_PATH;

pub const DOWNLOAD_SUBCOMMAND: &str = "download";
pub fn cli() -> Command {
    Command::new(DOWNLOAD_SUBCOMMAND)
        .about("Download the OpenWebText dataset from Hugging Face.")
        .arg(Arg::new("subsetCount")
            .short('s').long("subset-count")
            .help("The number of subsets to download. Each subset will extract to roughly 2.7GB! The data contains 21 subsets in total, equating to 55.2GB of data.")
            .default_value("1")
            .value_parser(parse_subset_count))
}

fn parse_subset_count(value: &str) -> Result<usize, String> {
    let parsed_value = value.parse::<usize>();
    match parsed_value {
        Ok(count) => {
            if count > 0 && count <= 21 {
                Ok(count)
            } else {
                Err("subsetCount must be a positive integer between 1 and 21.".to_string())
            }
        }
        Err(_) => Err("subsetCount must be a positive integer.".to_string())
    }
}

fn get_subset_url(subset_index: usize) -> String {
    format!("https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset{:02}.tar", subset_index)
}

pub async fn download(subsets: usize) {
    println!("Downloading {} subsets of the OpenWebText dataset to {}", subsets, crate::DATA_PATH.display());

    let client = reqwest::Client::new();
    let data_path = DATA_PATH.join("data");

    let multi_progress = Arc::new(indicatif::MultiProgress::new());

    // Asynchronously download each subset
    let mut set = JoinSet::new();
    for i in 0..subsets {
        set.spawn(download_subset(i, subsets, client.clone(), data_path.clone(), multi_progress.clone()));
    }

    set.join_all().await;

    println!("Finished downloading {} subsets of the OpenWebText dataset to {}", subsets, crate::DATA_PATH.display());
}

async fn download_subset(index: usize, subsets: usize, client: reqwest::Client, data_path: PathBuf, multi_progress: Arc<indicatif::MultiProgress>) {
    let progress_bar = multi_progress.add(
        indicatif::ProgressBar::new(1)
            .with_prefix(format!("Subset {:02}/{}", index + 1, subsets))
            .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len} | {msg}").unwrap())
    );
        
    // First, download the actual .tar file and show a progress bar
    let subset_filename = format!("subsets/urlsf_subset{:02}.tar", index);
    let subset_path = data_path.join(subset_filename);
    if !subset_path.parent().unwrap().exists() {
        tokio::fs::create_dir_all(subset_path.parent().unwrap()).await.unwrap();
    }

    if !subset_path.exists() {
        let subset_url = get_subset_url(index);
        progress_bar.set_message("Downloading...");

        let response = client.head(&subset_url).send().await.unwrap();
        let content_length = response.headers().get("content-length").unwrap().to_str().unwrap().parse::<u64>().unwrap();
        progress_bar.set_length(content_length);
    
        let mut file = tokio::fs::File::create(&subset_path).await.unwrap();
        let mut stream = client.get(&subset_url).send().await.unwrap().bytes_stream();
    
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            file.write_all(&chunk).await.unwrap();
            progress_bar.inc(chunk.len() as u64);
        }
    
        progress_bar.finish_with_message(format!("Downloaded subset to {}", subset_path.display()));
    
        file.sync_all().await.unwrap();
    } else {
        progress_bar.abandon_with_message(format!("Subset already exists at {}", subset_path.display()));
    }

    // Then, extract the .tar file
    let extract_path = data_path.join("extracted").join(format!("subset{:02}", index));

    if !extract_path.exists() {
        tokio::fs::create_dir_all(&extract_path).await.unwrap();
        progress_bar.set_message("Extracting subset...");

        let tar = tokio::fs::File::open(&subset_path).await.unwrap();
        let mut tar: tokio_tar::Archive<tokio::fs::File> = tokio_tar::Archive::new(tar);
        let mut entries = tar.entries().unwrap();

        let mut files: Vec<PathBuf> = vec![];
        while let Some(entry) = entries.next().await {
            let path: PathBuf = entry.as_ref().unwrap().path().unwrap().into();
            let path = extract_path.join(path.file_name().unwrap());
            if path.is_dir() {
                break;
            }
            if !path.extension().unwrap().eq("xz") {
                continue;
            }

            entry.unwrap().unpack(&path).await.unwrap();

            files.push(path);
        }

        progress_bar.set_length(files.len() as u64);
        progress_bar.set_position(0);

        progress_bar.set_message("Extracting subset child xz files...");

        // For each .xz file, extract it
        for path in files {
            let mut content = Vec::new();
            std::fs::File::open(&path)
                .unwrap()
                .read_to_end(&mut content)
                .unwrap();

            let mut decoder = xz2::read::XzDecoder::new(&content[..]);
            let mut output_tar_data = Vec::new();
            decoder.read_to_end(&mut output_tar_data).unwrap();

            let mut tar = tokio_tar::Archive::new(std::io::Cursor::new(output_tar_data));
            let mut entries = tar.entries().unwrap();
            
            while let Some(entry) = entries.next().await {
                let path: PathBuf = entry.as_ref().unwrap().path().unwrap().into();
                let path = extract_path.join(path.file_name().unwrap());
                if path.is_dir() {
                    break;
                }

                entry.unwrap().unpack(&path).await.unwrap();
            }

            // Delete the .xz file
            tokio::fs::remove_file(&path).await.unwrap();

            progress_bar.inc(1);
        }

        progress_bar.finish_with_message(format!("Extracted subset to {}", extract_path.display()));
    } else {
        progress_bar.abandon_with_message(format!("Subset already extracted to {}", extract_path.display()));
    }
}