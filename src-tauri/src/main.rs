use tauri::{AppHandle, Emitter, State};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::{interval, Duration};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

type Result<T> = std::result::Result<T, String>;

// ---------- PROVIDERS ----------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    // Locales
    LmStudio,
    Ollama,
    LocalAi,
    Custom,
    // Nube
    OpenAi,
    Groq,
    Gemini,
    Anthropic,
    Mistral,
    Together,
    OpenRouter,
}

impl LlmProvider {
    fn default_base_url(&self) -> &'static str {
        match self {
            LlmProvider::LmStudio  => "http://localhost:1234/v1",
            LlmProvider::Ollama    => "http://localhost:11434/v1",
            LlmProvider::LocalAi   => "http://localhost:8080/v1",
            LlmProvider::Custom    => "http://localhost:8000/v1",
            LlmProvider::OpenAi    => "https://api.openai.com/v1",
            LlmProvider::Groq      => "https://api.groq.com/openai/v1",
            LlmProvider::Gemini    => "https://generativelanguage.googleapis.com/v1beta/openai",
            LlmProvider::Anthropic => "https://api.anthropic.com/v1",
            LlmProvider::Mistral   => "https://api.mistral.ai/v1",
            LlmProvider::Together  => "https://api.together.xyz/v1",
            LlmProvider::OpenRouter => "https://openrouter.ai/api/v1",
        }
    }

    fn default_model(&self) -> &'static str {
        match self {
            LlmProvider::LmStudio  => "local-model",
            LlmProvider::Ollama    => "llama3.2",
            LlmProvider::LocalAi   => "gpt-4",
            LlmProvider::Custom    => "local-model",
            LlmProvider::OpenAi    => "gpt-4o-mini",
            LlmProvider::Groq      => "llama-3.3-70b-versatile",
            LlmProvider::Gemini    => "gemini-2.0-flash",
            LlmProvider::Anthropic => "claude-3-5-haiku-20241022",
            LlmProvider::Mistral   => "mistral-small-latest",
            LlmProvider::Together  => "meta-llama/Llama-3-8b-chat-hf",
            LlmProvider::OpenRouter => "openai/gpt-4o-mini",
        }
    }

    fn requires_api_key(&self) -> bool {
        matches!(
            self,
            LlmProvider::OpenAi
                | LlmProvider::Groq
                | LlmProvider::Gemini
                | LlmProvider::Anthropic
                | LlmProvider::Mistral
                | LlmProvider::Together
                | LlmProvider::OpenRouter
        )
    }

    fn is_anthropic(&self) -> bool {
        *self == LlmProvider::Anthropic
    }
}

impl Default for LlmProvider {
    fn default() -> Self {
        LlmProvider::LmStudio
    }
}

// ---------- CONFIG ----------

#[derive(Serialize, Deserialize, Clone)]
struct AppConfig {
    repo_path: String,
    auto_commit_enabled: bool,
    interval_minutes: u64,
    auto_start: bool,
    provider: LlmProvider,
    llm_base_url: String,
    llm_model_name: String,
    llm_api_key: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        let provider = LlmProvider::default();
        Self {
            repo_path: String::new(),
            auto_commit_enabled: false,
            interval_minutes: 30,
            auto_start: false,
            llm_base_url: provider.default_base_url().to_string(),
            llm_model_name: provider.default_model().to_string(),
            llm_api_key: String::new(),
            provider,
        }
    }
}

// ---------- STATE ----------

struct AppState {
    config: Arc<Mutex<AppConfig>>,
    timer_running: Arc<Mutex<bool>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            config: Arc::new(Mutex::new(AppConfig::default())),
            timer_running: Arc::new(Mutex::new(false)),
        }
    }
}

// ---------- ESTRUCTURAS OPENAI-COMPAT ----------

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

// Anthropic usa un formato diferente
#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    system: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

// ---------- SYSTEM PROMPT ----------

const SYSTEM_PROMPT: &str = r#"
You are an expert Git commit message generator.
Analyze the provided git diff and generate a concise, conventional commit message.
Format: <type>: <description>
Types: feat, fix, docs, style, refactor, test, chore.
Example: feat: add multi-provider LLM support
IMPORTANT: Respond ONLY with the commit message. No quotes, no explanations, no markdown.
"#;

// ---------- LÓGICA DE LLAMADA AL LLM ----------

async fn call_llm(
    provider: &LlmProvider,
    base_url: &str,
    model: &str,
    api_key: &str,
    user_content: &str,
) -> Result<String> {
    let client = Client::new();

    // Anthropic tiene API propia (no es OpenAI-compatible en autenticación ni formato)
    if provider.is_anthropic() {
        let endpoint = if base_url.ends_with('/') {
            format!("{}messages", base_url)
        } else {
            format!("{}/messages", base_url)
        };

        let request_body = AnthropicRequest {
            model: model.to_string(),
            system: SYSTEM_PROMPT.to_string(),
            max_tokens: 256,
            messages: vec![Message {
                role: "user".to_string(),
                content: user_content.to_string(),
            }],
        };

        let response = client
            .post(&endpoint)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Connection to Anthropic failed: {}", e))?;

        if !response.status().is_success() {
            let err = response.text().await.unwrap_or_default();
            return Err(format!("Anthropic API error: {}", err));
        }

        let parsed: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse Anthropic response: {}", e))?;

        return parsed
            .content
            .into_iter()
            .next()
            .map(|c| c.text)
            .ok_or_else(|| "Anthropic returned no content".to_string());
    }

    // Todos los demás proveedores usan formato OpenAI-compatible
    let endpoint = if base_url.ends_with('/') {
        format!("{}chat/completions", base_url)
    } else {
        format!("{}/chat/completions", base_url)
    };

    let request_body = ChatCompletionRequest {
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

    let mut req = client.post(&endpoint).json(&request_body);

    // Gemini usa ?key= en vez de Bearer, pero también acepta Bearer desde la URL v1beta/openai
    // Para simplificar usamos Bearer en todos (Gemini lo acepta con este endpoint)
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", api_key));
    }

    // OpenRouter requiere headers adicionales opcionales
    if *provider == LlmProvider::OpenRouter {
        req = req
            .header("HTTP-Referer", "https://github.com/auto-commit-tauri")
            .header("X-Title", "Auto Commit");
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Connection to {} failed: {}", base_url, e))?;

    if !response.status().is_success() {
        let err = response.text().await.unwrap_or_default();
        return Err(format!("LLM API error: {}", err));
    }

    let parsed: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "LLM returned no content".to_string())
}

// ---------- LÓGICA PRINCIPAL DE COMMIT ----------

async fn run_commit_internal(
    path: &str,
    provider: &LlmProvider,
    base_url: &str,
    model: &str,
    api_key: &str,
) -> Result<String> {
    // 1. Comprobar si hay cambios
    let status_output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(path)
        .output()
        .map_err(|e| format!("Git status error: {}", e))?;

    let status_text = String::from_utf8_lossy(&status_output.stdout);
    if status_text.trim().is_empty() {
        return Ok("No changes to commit".into());
    }

    // 2. Stage all changes
    Command::new("git")
        .args(["add", "."])
        .current_dir(path)
        .status()
        .map_err(|e| format!("Git add error: {}", e))?;

    // 3. Obtener diff
    let diff_output = Command::new("git")
        .args(["diff", "--cached"])
        .current_dir(path)
        .output()
        .map_err(|e| format!("Git diff error: {}", e))?;

    let diff_content = String::from_utf8_lossy(&diff_output.stdout);

    // Recortar diff (modelos cloud pueden manejar más contexto)
    let max_diff = if provider.requires_api_key() { 16_000 } else { 8_000 };
    let diff_text = if diff_content.len() > max_diff {
        format!("(Diff truncated to {} chars)...\n{}", max_diff, &diff_content[..max_diff])
    } else {
        diff_content.to_string()
    };

    let user_prompt = format!("Generate a commit message for these changes:\n\n{}", diff_text);

    // 4. Llamar al LLM
    let raw_message = call_llm(provider, base_url, model, api_key, &user_prompt).await?;

    let clean_message = raw_message
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();

    // 5. Commit
    Command::new("git")
        .args(["commit", "-m", &clean_message])
        .current_dir(path)
        .status()
        .map_err(|e| format!("Git commit error: {}", e))?;

    // 6. Push
    Command::new("git")
        .arg("push")
        .current_dir(path)
        .status()
        .map_err(|e| format!("Git push error: {}", e))?;

    Ok(clean_message)
}

// ---------- HELPERS ----------

fn get_config_path() -> Result<PathBuf> {
    let mut path = dirs::config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    path.push("auto-commit-app");
    fs::create_dir_all(&path)
        .map_err(|e| format!("Failed to create config directory: {}", e))?;
    path.push("config.json");
    Ok(path)
}

// ---------- COMANDOS TAURI ----------

#[tauri::command]
async fn run_commit(path: String, state: State<'_, AppState>) -> Result<String> {
    let (provider, base_url, model, api_key) = {
        let config = state.config.lock().map_err(|e| e.to_string())?;
        (
            config.provider.clone(),
            config.llm_base_url.clone(),
            config.llm_model_name.clone(),
            config.llm_api_key.clone(),
        )
    };
    run_commit_internal(&path, &provider, &base_url, &model, &api_key).await
}

#[tauri::command]
async fn save_config(config: AppConfig, state: State<'_, AppState>) -> Result<()> {
    let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
    *app_config = config.clone();

    let config_path = get_config_path()?;
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;
    fs::write(&config_path, config_json)
        .map_err(|e| format!("Failed to save config: {}", e))?;
    Ok(())
}

#[tauri::command]
async fn get_config(state: State<'_, AppState>) -> Result<AppConfig> {
    let config = state.config.lock().map_err(|e| e.to_string())?;
    Ok(config.clone())
}

#[tauri::command]
async fn load_config_from_file(state: State<'_, AppState>) -> Result<AppConfig> {
    let config_path = get_config_path()?;
    if config_path.exists() {
        let config_str = fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        let config: AppConfig = serde_json::from_str(&config_str)
            .unwrap_or_default();
        let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
        *app_config = config.clone();
        Ok(config)
    } else {
        Ok(AppConfig::default())
    }
}

#[tauri::command]
async fn get_provider_defaults(provider: LlmProvider) -> Result<(String, String)> {
    Ok((
        provider.default_base_url().to_string(),
        provider.default_model().to_string(),
    ))
}

#[tauri::command]
async fn start_auto_commit(
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<()> {
    let (interval_minutes, repo_path, provider, base_url, model, api_key) = {
        let config = state.config.lock().map_err(|e| e.to_string())?;
        (
            config.interval_minutes,
            config.repo_path.clone(),
            config.provider.clone(),
            config.llm_base_url.clone(),
            config.llm_model_name.clone(),
            config.llm_api_key.clone(),
        )
    };

    if interval_minutes == 0 {
        return Err("Interval must be at least 1 minute.".into());
    }

    let mut timer_running = state.timer_running.lock().map_err(|e| e.to_string())?;
    if *timer_running {
        return Err("Timer is already running".into());
    }
    *timer_running = true;
    drop(timer_running);

    let timer_running_arc = Arc::clone(&state.timer_running);
    tauri::async_runtime::spawn(async move {
        let mut interval_timer = interval(Duration::from_secs(interval_minutes * 60));
        loop {
            interval_timer.tick().await;

            let should_stop = {
                let running = timer_running_arc.lock().unwrap();
                !*running
            };
            if should_stop {
                break;
            }

            match run_commit_internal(&repo_path, &provider, &base_url, &model, &api_key).await {
                Ok(msg) => {
                    if msg != "No changes to commit" {
                        let _ = app_handle.emit("commit-status", msg);
                    }
                }
                Err(e) => {
                    let _ = app_handle.emit("commit-error", e);
                }
            }
        }
    });

    Ok(())
}

#[tauri::command]
async fn stop_auto_commit(state: State<'_, AppState>) -> Result<()> {
    let mut timer_running = state.timer_running.lock().map_err(|e| e.to_string())?;
    *timer_running = false;
    Ok(())
}

#[tauri::command]
async fn select_directory() -> Result<String> {
    Err("Please manually paste the path for now".into())
}

#[tauri::command]
async fn test_connection(
    provider: LlmProvider,
    base_url: String,
    model: String,
    api_key: String,
) -> Result<String> {
    call_llm(
        &provider,
        &base_url,
        &model,
        &api_key,
        "Reply with exactly the word 'Connected'.",
    )
        .await
        .map(|_| "Connection successful!".to_string())
}

// ---------- MAIN ----------

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            run_commit,
            save_config,
            get_config,
            load_config_from_file,
            get_provider_defaults,
            start_auto_commit,
            stop_auto_commit,
            select_directory,
            test_connection
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}