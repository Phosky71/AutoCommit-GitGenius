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
You are an expert developer and a strict Git commit message generator.
Analyze the diff and generate a comprehensive commit message using the Conventional Commits standard.

RULES:
1. The FIRST LINE must be the title: <type>(<scope>): <description> (under 72 characters).
2. Valid types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
3. The title description MUST focus on the WHY and the IMPACT of the change.
4. If multiple files/modules are changed, you MUST add a blank line after the title, followed by a bulleted list explaining what was changed in each file/module.
5. NEVER wrap the output in quotes, backticks, code blocks, or markdown (```).
6. NEVER output conversational text like "Here is your message".

EXPECTED OUTPUT FORMAT:
feat(settings): add git PAT input to enable silent background push

- src/config.rs: add git_token field to AppConfig state
- src/git.rs: inject token into git remote url to bypass credential helper
- js/settings.js: add UI input to securely save and load the token
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
    provider: &LlmProvider, base_url: &str, model: &str, api_key: &str, user_content: &str,
) -> Result<String> {
    call_llm_with_system(provider, base_url, model, api_key, SYSTEM_PROMPT, user_content).await
}

pub async fn call_llm_with_system(
    provider: &LlmProvider,
    base_url: &str,
    model: &str,
    api_key: &str,
    system_prompt: &str,
    user_content: &str,
) -> Result<String> {
    if provider.is_anthropic() {
        let endpoint = format!("{}/messages", base_url.trim_end_matches('/'));
        let body = AnthropicRequest {
            model: model.to_string(),
            system: system_prompt.to_string(),
            max_tokens: 256,
            messages: vec![Message { role: "user".to_string(), content: user_content.to_string() }],
        };
        let resp = HTTP_CLIENT.post(&endpoint)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body).send().await
            .map_err(|e| format!("Anthropic connection failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("Anthropic error (HTTP {}): {}", resp.status(), resp.text().await.unwrap_or_default()));
        }
        let raw_text = resp.text().await.map_err(|e| format!("Failed to read Anthropic response: {}", e))?;
        let parsed: AnthropicResponse = serde_json::from_str(&raw_text)
            .map_err(|e| format!("Parse error: {}\nRaw response: {}", e, raw_text))?;

        return parsed.content.into_iter().next().map(|c| c.text)
            .ok_or_else(|| "Anthropic returned empty content array".to_string());
    }

    let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = ChatCompletionRequest {
        model: model.to_string(),
        temperature: 0.3,
        max_tokens: Some(256),
        messages: vec![
            Message { role: "system".to_string(), content: system_prompt.to_string() },
            Message { role: "user".to_string(), content: user_content.to_string() },
        ],
    };

    let mut req = HTTP_CLIENT.post(&endpoint).json(&body);
    if !api_key.is_empty() { req = req.header("Authorization", format!("Bearer {}", api_key)); }

    let resp = req.send().await
        .map_err(|e| format!("Connection to {} failed: {}", base_url, e))?;

    if !resp.status().is_success() {
        return Err(format!("LLM API error (HTTP {}): {}", resp.status(), resp.text().await.unwrap_or_default()));
    }
    let raw_text = resp.text().await.map_err(|e| format!("Failed to read LLM response: {}", e))?;
    let parsed: ChatCompletionResponse = serde_json::from_str(&raw_text)
        .map_err(|e| format!("Parse error: {}\nRaw response: {}", e, raw_text))?;

    parsed.choices.into_iter().next().map(|c| c.message.content)
        .ok_or_else(|| "LLM returned no choices".to_string())
}

pub async fn ask_llm_if_ready(
    provider: &LlmProvider, base_url: &str, model: &str, api_key: &str, diff_content: &str,
) -> Result<bool> {
    let ai_system_prompt = "You are an extremely strict AI Git Assistant. Your ONLY job is to decide if a diff is ready to be committed. \
    Consider it NOT ready if: it has obvious unfinished lines, syntax errors, or is too trivial (e.g., just a console.log). \
    If it is a logical, coherent unit of work, answer strictly with 'YES'. Otherwise, answer strictly with 'NO'. Never explain your reasoning.";

    // Truncar para no quemar tokens con diffs masivos en la evaluación
    let max_diff = 4000;
    let diff_text = if diff_content.len() > max_diff {
        format!("(Diff truncated)...\n{}", &diff_content[..max_diff])
    } else {
        diff_content.to_string()
    };

    let user_prompt = format!("Analyze this diff:\n{}", diff_text);

    let response = call_llm_with_system(provider, base_url, model, api_key, ai_system_prompt, &user_prompt).await?;

    // Evaluamos de forma segura si respondió YES
    Ok(response.trim().to_uppercase().starts_with("YES"))
}

