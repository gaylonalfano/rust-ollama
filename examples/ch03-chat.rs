// NOTE: !! Chat API example from ollama-rs
// REF: https://github.com/pepperoni21/ollama-rs/blob/master/examples/chat_api_chatbot.rs

// region:       -- Modules

use futures::StreamExt;
use tokio::io::AsyncWriteExt;
use xp_ollama::constants::{DEFAULT_SYSTEM_PROMPT, MODEL};
use xp_ollama::gen::gen_stream_print;
use xp_ollama::Result;

use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, ChatMessageResponseStream, MessageRole};
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::completion::GenerationContext;
use ollama_rs::Ollama;

// endregion:    -- Modules

#[tokio::main]
async fn main() -> Result<()> {
    // NOTE: Default Ollama port is http://127.0.0.1:11434
    let ollama = Ollama::default();

    let prompts = &[
        "What is the best language? (be concise)",
        // "What is the second best language?",
        "Why is the sky red?",
        "What was my last question?",
    ];

    let system_message = ChatMessage::new(MessageRole::System, DEFAULT_SYSTEM_PROMPT.to_string());

    let mut thread_messages: Vec<ChatMessage> = vec![system_message];

    for prompt in prompts {
        println!("\n->> prompt: {prompt}");

        let prompt_message = ChatMessage::new(MessageRole::User, prompt.to_string());
        thread_messages.push(prompt_message);

        // NOTE: We need an Owned copy of messages, so we use clone().
        let chat_request = ChatMessageRequest::new(MODEL.to_string(), thread_messages.clone());

        let message_content = run_chat_req(&ollama, chat_request).await?;

        if let Some(content) = message_content {
            let assistant_message = ChatMessage::new(MessageRole::Assistant, content);
            thread_messages.push(assistant_message);
        }
    }

    Ok(())
}

// NOTE: TIP! - Return a simple Result<Option<String>> while still learning the api and exploring.
pub async fn run_chat_req(
    ollama: &Ollama,
    chat_request: ChatMessageRequest,
) -> Result<Option<String>> {
    let mut stream: ChatMessageResponseStream =
        ollama.send_chat_messages_stream(chat_request).await?;
    // let mut stream = ollama.send_chat_messages_stream(chat_request).await?;

    let mut stdout = tokio::io::stdout();
    let mut char_count = 0;
    // NOTE:!! Store each message chunk we receive so we can rebuild the complete/final
    // message. This API doesn't return the full message at the end like
    // the ch02-context final_data has.
    // NOTE: !! The ollama-rs docs uses a simple String to build the final message
    // using response += assistant_message.content.as_str()
    // REF: https://github.com/pepperoni21/ollama-rs/blob/master/examples/chat_api_chatbot.rs
    // let mut response = String::new();
    let mut current_asst_msg_elems: Vec<String> = Vec::new();

    while let Some(Ok(res)) = stream.next().await {
        if let Some(assistant_message) = res.message {
            // Get the content (partial, chunk, etc)
            let message_content = assistant_message.content;

            // Poor man's wrapping
            char_count = 0;
            if char_count > 80 {
                stdout.write_all(b"\n").await?;
                char_count = 0;
            }

            // Write output
            stdout.write_all(message_content.as_bytes()).await?;
            stdout.flush().await?;

            // Add content chunk to our vector so we can rebuild complete message
            current_asst_msg_elems.push(message_content);
        }

        // Get final response data if available
        if let Some(_final_res) = res.final_data {
            // NOTE:ChatMessageFinalResponseData doesn't hold the final completed
            // assistant response so we don't use it, hence the "_final_res" naming.
            // Q: Is it here we build the full message and store locally?
            // A: Yes, we build it here but don't store locally...
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;

            // Q: How to take a Vec<String> and build into a single String?
            // A: Use Vec.join(" ") or Vec.iter().fold(String::new(), |acc, elem| acc += elem)
            let assistant_content = current_asst_msg_elems.join("");

            // U: Trying to save the completed response locally
            // let assistant_content_path = ".ch03-data/asst_final.json";
            // simple_fs::ensure_file_dir(assistant_content_path)?;
            // simple_fs::save_json(assistant_content_path, &assistant_content)?;

            return Ok(Some(assistant_content));
        }
    }

    // new line
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;

    // Didn't get the final_data
    Ok(None)
}
