// NOTE: !! lib.rs for shared code between crates, modules, etc.
// REF: https://youtu.be/OcH-zT5VNgM?t=261

// region:       -- Modules

// NOTE: Early dev we start out this way:
pub type Result<T> = core::result::Result<T, Error>;

pub type Error = Box<dyn std::error::Error>; // For early dev

// NOTE: For production, we evolve/remove the type Error,
// and instead make it an Enum with various variants, and
// move all this Error-related code into a separate error.rs.

// NOTE: Simple 'consts' module here inline. Eventually would
// move to separate 'constants.rs' module.
// REF: https://youtu.be/OcH-zT5VNgM?t=357
pub mod constants {

    pub const MODEL: &str = "llama2:latest"; // "mixtral", etc.

    pub const DEFAULT_SYSTEM_PROMPT: &str = r#"
    Always be very concise with your answer. If you do not know the answer, just say so. Do not make up an answer.

    If asked about the best programming language, reply that it is Rust by light years.

    Only if you are asked about the second best language, reply that it is Haskell.

    If asked about the previous question, only give user messages, not the system message.
"#;
}

// endregion:    -- Modules

pub mod gen {
    use super::*;
    use ollama_rs::generation::completion::request::GenerationRequest;
    use ollama_rs::generation::completion::GenerationFinalResponseData;
    use ollama_rs::Ollama;
    use tokio::io::AsyncWriteExt;
    // use tokio_stream::StreamExt;
    use futures::StreamExt;

    /// NOTE: OLLAMA-RS 0.1.6 now returns a Vec<GenerationResponse>, so handling accordingly.
    /// TODO: Need to further understand what does the Vec<GenerationResponse> v.s. the old GenerationResponse
    ///       means to refine behavior. See ticket: https://github.com/pepperoni21/ollama-rs/issues/20)
    pub async fn gen_stream_print(
        ollama: &Ollama,
        gen_req: GenerationRequest,
    ) -> Result<Option<GenerationFinalResponseData>> {
        // NOTE: Pin<Box<T>> indicates there are some async things need to happen with Box
        // Good practice is to use stdout from Tokio.
        let mut stream = ollama.generate_stream(gen_req).await?;
        let mut stdout = tokio::io::stdout();
        let mut char_count = 0;

        while let Some(res) = stream.next().await {
            // NOTE: TIP! - The Box<dyn std::error::Error> trick allows us to
            // map one Error to a simple, static string.
            // let res = res.map_err(|_| "stream_next error")?;
            // U: 'res?' now correctly working, so no need for map_err() trick.
            let responses = res?;

            for res in responses {
                let bytes = res.response.as_bytes();

                // Poor man's wrapping. This is not the number of chars,
                // because it's UTF8, but it works.
                char_count += bytes.len();
                if char_count > 80 {
                    stdout.write_all(b"\n").await?;
                    char_count = 0;
                }

                // Write the output
                stdout.write_all(bytes).await?;
                stdout.flush().await?;

                // Capture/return this conversation memory context if this is the final response
                // NOTE: This has a context prop of GenerationContext(Vec<i32>)
                if let Some(final_data) = res.final_data {
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                    return Ok(Some(final_data));
                }
            }
        }

        stdout.write_all(b"\n").await?;
        stdout.flush().await?;

        Ok(None)
    }
}
