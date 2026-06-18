use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::config::{LlmProvider, Result};

// AÑADIDO: Configurar un timeout global para el cliente HTTP.
// Si un LLM local se queda colgado, no bloqueará el hilo de Tokio de Tauri indefinidamente.
pub static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(45))
        .build()
        .expect("Failed to build HTTP client")
});

pub const SYSTEM_PROMPT: &str = r#"
You are a strict Git commit message generator.
Analyze the diff and generate a SINGLE LINE commit message using the Conventional Commits standard.

RULES:
1. Format: <type>(<scope>): <description>
2. Valid types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
3. The description MUST be in lowercase, imperative mood ("add", not "added" or "adds"), and under 72 characters.
4. NEVER wrap the output in quotes, backticks, code blocks, or markdown.
5. NEVER output any conversational text, greetings, or explanations.

BAD OUTPUT: Here is your message: `feat: add login`
GOOD OUTPUT: feat(auth): implement JWT login
"#;

// ---------- REQUEST / RESPONSE STRUCTS ----------

#[derive(Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: Message,
}

#[derive(Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub system: String,
}

#[derive(Deserialize, Debug)]
pub struct AnthropicResponse {
    pub content: Vec<AnthropicContent>,
}

#[derive(Deserialize, Debug)]
pub struct AnthropicContent {
    pub text: String,
}

// ---------- LLM CALL ----------

pub async fn call_llm(
    provider: &LlmProvider,
    base_url: &str,
    model: &str,
    api_key: &str,
    user_content: &str,
) -> Result<String> {
    if provider.is_anthropic() {
        let endpoint = format!("{}/messages", base_url.trim_end_matches('/'));
        let body = AnthropicRequest {
            model: model.to_string(),
            system: SYSTEM_PROMPT.to_string(),
            max_tokens: 256,
            messages: vec![Message {
                role: "user".to_string(),
                content: user_content.to_string(),
            }],
        };
        let resp = HTTP_CLIENT
            .post(&endpoint)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Anthropic connection failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Anthropic error (HTTP {}): {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            ));
        }

        // BUG FIX: Leer la respuesta como texto primero para tenerla de contexto en caso de error de parseo
        let raw_text = resp.text().await.map_err(|e| format!("Failed to read Anthropic response: {}", e))?;

        let parsed: AnthropicResponse = serde_json::from_str(&raw_text)
            .map_err(|e| format!("Parse error: {}\nRaw response: {}", e, raw_text))?;

        return parsed
            .content
            .into_iter()
            .next()
            .map(|c| c.text)
            .ok_or_else(|| "Anthropic returned empty content array".to_string());
    }

    let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = ChatCompletionRequest {
        model: model.to_string(),
        temperature: 0.3,
        max_tokens: Some(256),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: user_content.to_string(),
            },
        ],
    };

    let mut req = HTTP_CLIENT.post(&endpoint).json(&body);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", api_key));
    }
    if *provider == LlmProvider::OpenRouter {
        req = req
            .header("HTTP-Referer", "https://github.com/auto-commit-tauri") // Idealmente, cambia a la URL de tu repo real
            .header("X-Title", "Auto Commit");
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Connection to {} failed: {}", base_url, e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "LLM API error (HTTP {}): {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }

    // BUG FIX: Leer texto en crudo para mejorar los mensajes de error
    let raw_text = resp.text().await.map_err(|e| format!("Failed to read LLM response: {}", e))?;

    let parsed: ChatCompletionResponse = serde_json::from_str(&raw_text)
        .map_err(|e| format!("Parse error: {}\nRaw response: {}", e, raw_text))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "LLM returned no choices".to_string())
}