// NOTE: !! It's in '/examples' that we have main()
// to test our code. We use lib.rs for SHARED code.

// region:       -- Modules

use xp_ollama::constants::{DEFAULT_SYSTEM_PROMPT, MODEL};
use xp_ollama::gen::gen_stream_print;
use xp_ollama::Result;

use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::Ollama;
// endregion:    -- Modules

#[tokio::main]
async fn main() -> Result<()> {
    // NOTE: Default Ollama port is 11434
    let ollama = Ollama::default();

    let model = MODEL.to_string();
    // let prompt = "What is the best programming language?".to_string();
    let prompt = "Summarize the khalifates in the years of early Islam".to_string();

    let gen_req = GenerationRequest::new(model, prompt).system(DEFAULT_SYSTEM_PROMPT.to_string());

    // -- Single Response Generation
    // let res = ollama.generate(gen_req).await?;
    // println!("->> res {}", res.response);

    // -- Stream Response Generation
    gen_stream_print(&ollama, gen_req).await?;

    Ok(())
}
