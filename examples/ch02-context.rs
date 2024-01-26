// NOTE: !! It's in '/examples' that we have main()
// to test our code. We use lib.rs for SHARED code.

// region:       -- Modules

use xp_ollama::constants::{DEFAULT_SYSTEM_PROMPT, MODEL};
use xp_ollama::gen::gen_stream_print;
use xp_ollama::Result;

use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::completion::GenerationContext;
use ollama_rs::Ollama;

// endregion:    -- Modules

#[tokio::main]
async fn main() -> Result<()> {
    // NOTE: Default Ollama port is http://127.0.0.1:11434
    let ollama = Ollama::default();

    let prompts = &[
        "Why is the sky red? (be concise)",
        "What was my first question?",
    ];

    // Save the conversation memory
    let mut last_ctx: Option<GenerationContext> = None;

    for prompt in prompts {
        println!("\n->> prompt: {prompt}");
        let mut gen_req = GenerationRequest::new(MODEL.to_string(), prompt.to_string());
        // NOTE: We get the last context when the request is completed. We need to set/configure
        // the context on our gen_req if so BEFORE we send the actual request.
        // NOTE: GenerationRequest has a 'context: Option<GenerationContext>' property, so we
        // make the gen_req mutable and update/set the 'context' value if we have last_ctx.
        // NOTE: last_ctx.take() ->> last_ctx is now None.
        // REF: https://youtu.be/OcH-zT5VNgM?t=1475
        if let Some(last_ctx) = last_ctx.take() {
            gen_req = gen_req.context(last_ctx);
        }

        let final_data = gen_stream_print(&ollama, gen_req).await?;

        // Store the context if we are at the end of the request
        if let Some(final_data) = final_data {
            // Q: Okay, we have an Option, so what to do with it?
            // NOTE: !! A: Got it! We want to store the conversation context between
            // all of the prompts we send off. So, we take the last_ctx and update
            // the gen_req.context(last_ctx) BEFORE we send the real request. Then,
            // after the request completes, we grab the new/last context of this
            // current request, and then use it as the context for the NEXT prompt!
            last_ctx = Some(final_data.context); // GenerationContext<Vec<i32>>

            // NOTE: Save for debug. We want to get an idea of the context size
            // so we create a new local hidden file (don't track with git). We can
            // save as JSON since GenerationContext impls Serialize.
            // REF: https://youtu.be/OcH-zT5VNgM?t=1845
            let ctx_file_path = ".ch02-data/ctx.json";
            simple_fs::ensure_file_dir(ctx_file_path)?;
            simple_fs::save_json(ctx_file_path, &last_ctx)?;
        }
    }

    Ok(())
}
