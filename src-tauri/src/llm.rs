use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::{LlmProvider, Result};

pub static HTTP_CLIENT: Lazy<Client> = Lazy::new(Client::new);

pub const SYSTEM_PROMPT: &str = r#"
You are an expert Git commit message generator.
Analyze the provided git diff and generate a concise, conventional commit message.
Format: <type>(<optional scope>): <description>
Types: feat, fix, docs, style, refactor, test, chore.
Example: feat: add multi-provider LLM support
IMPORTANT: Respond ONLY with the commit message. No quotes, no explanations, no markdown.
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

#[derive(Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
pub struct AnthropicResponse {
    pub content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
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
                "Anthropic error: {}",
                resp.text().await.unwrap_or_default()
            ));
        }
        let parsed: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;
        return parsed
            .content
            .into_iter()
            .next()
            .map(|c| c.text)
            .ok_or_else(|| "Anthropic returned no content".to_string());
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
            .header("HTTP-Referer", "https://github.com/auto-commit-tauri")
            .header("X-Title", "Auto Commit");
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Connection to {} failed: {}", base_url, e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "LLM API error: {}",
            resp.text().await.unwrap_or_default()
        ));
    }
    let parsed: ChatCompletionResponse = resp
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;
    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "LLM returned no content".to_string())
}