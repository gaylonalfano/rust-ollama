use std::{fs, path::Path};

use simple_fs::{ensure_dir, read_to_string};
use xp_ollama::constants::{DEFAULT_SYSTEM_PROMPT, MODEL};
use xp_ollama::Result;

use ollama_rs::Ollama;

const MOCK_DIR: &str = "_mock-data";
const CH04_DIR: &str = ".ch04-data";
const DATA_FILE: &str = "for-embeddings.txt";

#[tokio::main]
async fn main() -> Result<()> {
    let ollama = Ollama::default();

    // Make sure we have a local dir
    ensure_dir(CH04_DIR)?;

    let txt = read_to_string(Path::new(MOCK_DIR).join(DATA_FILE))?;
    let splits = simple_text_splitter(&txt, 500)?;

    println!("->>      splits count: {}", splits.len());

    for (i, seg) in splits.into_iter().enumerate() {
        println!();
        // -- Save the text splits into TXT files
        let file_name = format!("ch04-embeddings-{:0>2}.txt", i);
        let file_path = Path::new(CH04_DIR).join(file_name);
        fs::write(file_path, &seg)?;

        println!("->>      text length: {}", seg.len());

        // -- Generate embeddings from segment/chunk
        let embeddings_response = ollama
            .generate_embeddings(MODEL.to_string(), seg, None)
            .await?;

        // Display the size of the embeddings
        println!(
            "->> embeddings length: {}",
            embeddings_response.embeddings.len()
        );

        // -- Save as JSON
        let file_name = format!("ch04-embeddings-{:0>2}.json", i);
        let file_path = Path::new(CH04_DIR).join(file_name);
        // Q: How to write Vec<f64> to JSON?
        // A: Use simple_fs::save_json() OR serde_json::to_writer()
        simple_fs::save_json(file_path, &embeddings_response.embeddings)?;

        // -- Save as Binary (big endian f64) for comparison
        // NOTE: It's these binaries that we would store in a Vector DB
        let file_name = format!("ch04-embeddings-{:0>2}.be-f64.bin", i);
        let file_path = Path::new(CH04_DIR).join(file_name);
        simple_fs::save_be_f64(file_path, &embeddings_response.embeddings)?;
    }

    Ok(())
}

/// A SILLY text splitter on "char" num only for exploration ONLY.
fn simple_text_splitter(txt: &str, num: u32) -> Result<Vec<String>> {
    let mut result = Vec::new();
    let mut last = 0;
    let mut count = 0;

    // NOTE: We want to go through the slice of text up to the 'num' value,
    // and append each slice/chunk to our result Vec.
    for (idx, _) in txt.char_indices() {
        count += 1;
        // // My attempt:
        // if count == num {
        //     // Append current slice/chunk to result Vec
        //     let current_split = &txt[last..idx];
        //     result.push(current_split.to_string());
        //
        //     // Update 'last' and reset 'count'
        //     last = idx;
        //     count = 0;
        // }

        // Alternative:
        if count == num {
            result.push(&txt[last..idx + 1]);
            last = idx + 1;
            count = 0;
        }
    }

    // Handle any remaining characters
    if last < txt.len() {
        result.push(&txt[last..]);
    }

    Ok(result.into_iter().map(String::from).collect())
}
