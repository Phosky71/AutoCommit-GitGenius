use tauri::{AppHandle, Emitter, State, Manager};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::{interval, Duration};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, String>;

// ---------- PROVIDERS ----------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    LmStudio,
    Ollama,
    LocalAi,
    Custom,
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
            LlmProvider::LmStudio   => "http://localhost:1234/v1",
            LlmProvider::Ollama     => "http://localhost:11434/v1",
            LlmProvider::LocalAi    => "http://localhost:8080/v1",
            LlmProvider::Custom     => "http://localhost:8000/v1",
            LlmProvider::OpenAi     => "https://api.openai.com/v1",
            LlmProvider::Groq       => "https://api.groq.com/openai/v1",
            LlmProvider::Gemini     => "https://generativelanguage.googleapis.com/v1beta/openai",
            LlmProvider::Anthropic  => "https://api.anthropic.com/v1",
            LlmProvider::Mistral    => "https://api.mistral.ai/v1",
            LlmProvider::Together   => "https://api.together.xyz/v1",
            LlmProvider::OpenRouter => "https://openrouter.ai/api/v1",
        }
    }

    fn default_model(&self) -> &'static str {
        match self {
            LlmProvider::LmStudio   => "local-model",
            LlmProvider::Ollama     => "llama3.2",
            LlmProvider::LocalAi    => "gpt-4",
            LlmProvider::Custom     => "local-model",
            LlmProvider::OpenAi     => "gpt-4o-mini",
            LlmProvider::Groq       => "llama-3.3-70b-versatile",
            LlmProvider::Gemini     => "gemini-2.0-flash",
            LlmProvider::Anthropic  => "claude-3-5-haiku-20241022",
            LlmProvider::Mistral    => "mistral-small-latest",
            LlmProvider::Together   => "meta-llama/Llama-3-8b-chat-hf",
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
    fn default() -> Self { LlmProvider::LmStudio }
}

// ---------- SMART COMMIT MODE ----------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SmartMode {
    Always,
    Smart,
    Never,
}

impl Default for SmartMode {
    fn default() -> Self { SmartMode::Smart }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DiffStats {
    files_changed: usize,
    insertions: usize,
    deletions: usize,
    is_significant: bool,
    estimated_tokens: usize,
}

fn analyze_diff(diff: &str, threshold_lines: u64) -> DiffStats {
    let mut insertions = 0usize;
    let mut deletions = 0usize;
    let mut files_changed = 0usize;

    for line in diff.lines() {
        if line.starts_with('+') && !line.starts_with("+++") { insertions += 1; }
        else if line.starts_with('-') && !line.starts_with("---") { deletions += 1; }
        else if line.starts_with("diff --git") { files_changed += 1; }
    }

    let total_lines = (insertions + deletions) as u64;
    let is_significant = total_lines >= threshold_lines || files_changed >= 3;
    // Rough token estimate: ~4 chars per token
    let estimated_tokens = diff.len() / 4;

    DiffStats { files_changed, insertions, deletions, is_significant, estimated_tokens }
}

fn generate_fallback_message(stats: &DiffStats) -> String {
    if stats.files_changed == 0 {
        return "chore: minor update".to_string();
    }
    if stats.deletions > stats.insertions * 2 {
        return format!("refactor: remove code across {} file(s)", stats.files_changed);
    }
    if stats.insertions > 0 && stats.deletions == 0 {
        return format!("feat: add new code in {} file(s)", stats.files_changed);
    }
    format!(
        "chore: update {} file(s) (+{} -{} lines)",
        stats.files_changed, stats.insertions, stats.deletions
    )
}

// ---------- MULTI-REPO ----------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepoEntry {
    pub id: String,
    pub path: String,
    pub interval_minutes: u64,
    pub enabled: bool,
    pub push_enabled: bool,
    pub push_branch: String,
    pub commit_prefix: String,
    pub last_commit_time: u64,
    pub cooldown_minutes: u64,
}

impl Default for RepoEntry {
    fn default() -> Self {
        Self {
            id: uuid_v4(),
            path: String::new(),
            interval_minutes: 30,
            enabled: false,
            push_enabled: true,
            push_branch: "origin/main".to_string(),
            commit_prefix: String::new(),
            last_commit_time: 0,
            cooldown_minutes: 5,
        }
    }
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().subsec_nanos();
    format!("{:x}-auto", t)
}

// ---------- COMMIT HISTORY ----------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CommitHistoryEntry {
    pub timestamp: u64,
    pub repo_path: String,
    pub message: String,
    pub used_llm: bool,
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub estimated_tokens: usize,
}

// ---------- CONFIG ----------

#[derive(Serialize, Deserialize, Clone)]
struct AppConfig {
    // Legacy single-repo (kept for compatibility)
    repo_path: String,
    auto_commit_enabled: bool,
    interval_minutes: u64,
    auto_start: bool,
    provider: LlmProvider,
    llm_base_url: String,
    llm_model_name: String,
    llm_api_key: String,
    smart_mode: SmartMode,
    smart_threshold_lines: u64,

    // New fields
    push_enabled: bool,
    push_branch: String,
    commit_prefix: String,
    cooldown_minutes: u64,

    // Multi-repo
    repos: Vec<RepoEntry>,

    // History
    commit_history: Vec<CommitHistoryEntry>,
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
            smart_mode: SmartMode::default(),
            smart_threshold_lines: 10,
            push_enabled: true,
            push_branch: "origin/main".to_string(),
            commit_prefix: String::new(),
            cooldown_minutes: 5,
            repos: Vec::new(),
            commit_history: Vec::new(),
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

// ---------- LLM STRUCTS ----------

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

const SYSTEM_PROMPT: &str = r#"
You are an expert Git commit message generator.
Analyze the provided git diff and generate a concise, conventional commit message.
Format: <type>(<optional scope>): <description>
Types: feat, fix, docs, style, refactor, test, chore.
Example: feat: add multi-provider LLM support
IMPORTANT: Respond ONLY with the commit message. No quotes, no explanations, no markdown.
"#;

// ---------- LLM CALL ----------

async fn call_llm(
    provider: &LlmProvider,
    base_url: &str,
    model: &str,
    api_key: &str,
    user_content: &str,
) -> Result<String> {
    let client = Client::new();

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
            messages: vec![Message { role: "user".to_string(), content: user_content.to_string() }],
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
        return parsed.content.into_iter().next()
            .map(|c| c.text)
            .ok_or_else(|| "Anthropic returned no content".to_string());
    }

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
            Message { role: "system".to_string(), content: SYSTEM_PROMPT.to_string() },
            Message { role: "user".to_string(), content: user_content.to_string() },
        ],
    };

    let mut req = client.post(&endpoint).json(&request_body);

    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", api_key));
    }

    if *provider == LlmProvider::OpenRouter {
        req = req
            .header("HTTP-Referer", "https://github.com/auto-commit-tauri")
            .header("X-Title", "Auto Commit");
    }

    let response = req.send().await
        .map_err(|e| format!("Connection to {} failed: {}", base_url, e))?;

    if !response.status().is_success() {
        let err = response.text().await.unwrap_or_default();
        return Err(format!("LLM API error: {}", err));
    }

    let parsed: ChatCompletionResponse = response.json().await
        .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

    parsed.choices.into_iter().next()
        .map(|c| c.message.content)
        .ok_or_else(|| "LLM returned no content".to_string())
}

// ---------- PUBLIC RESULT TYPES ----------

#[derive(Serialize, Deserialize, Clone)]
pub struct DiffStatsPublic {
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub estimated_tokens: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CommitResult {
    pub message: String,
    pub used_llm: bool,
    pub diff_stats: Option<DiffStatsPublic>,
}

// ---------- CORE COMMIT LOGIC ----------

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

async fn run_commit_internal(
    path: &str,
    provider: &LlmProvider,
    base_url: &str,
    model: &str,
    api_key: &str,
    smart_mode: &SmartMode,
    smart_threshold_lines: u64,
    push_enabled: bool,
    push_branch: &str,
    commit_prefix: &str,
    cooldown_minutes: u64,
    last_commit_time: u64,
    dry_run: bool,
) -> Result<CommitResult> {
    // Cooldown check
    if !dry_run && cooldown_minutes > 0 {
        let elapsed = now_unix().saturating_sub(last_commit_time);
        if elapsed < cooldown_minutes * 60 {
            return Ok(CommitResult {
                message: format!("Cooldown active: {}s remaining", cooldown_minutes * 60 - elapsed),
                used_llm: false,
                diff_stats: None,
            });
        }
    }

    // 1. Check changes
    let status_output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(path)
        .output()
        .map_err(|e| format!("Git status error: {}", e))?;

    let status_text = String::from_utf8_lossy(&status_output.stdout);
    if status_text.trim().is_empty() {
        return Ok(CommitResult {
            message: "No changes to commit".to_string(),
            used_llm: false,
            diff_stats: None,
        });
    }

    // 2. Stage all
    if !dry_run {
        Command::new("git")
            .args(["add", "."])
            .current_dir(path)
            .status()
            .map_err(|e| format!("Git add error: {}", e))?;
    }

    // 3. Get diff (for dry_run use unstaged diff)
    let diff_args = if dry_run {
        vec!["diff"]
    } else {
        vec!["diff", "--cached"]
    };
    let diff_output = Command::new("git")
        .args(&diff_args)
        .current_dir(path)
        .output()
        .map_err(|e| format!("Git diff error: {}", e))?;

    let diff_content = String::from_utf8_lossy(&diff_output.stdout);

    // 4. Analyze
    let stats = analyze_diff(&diff_content, smart_threshold_lines);
    let stats_public = DiffStatsPublic {
        files_changed: stats.files_changed,
        insertions: stats.insertions,
        deletions: stats.deletions,
        estimated_tokens: stats.estimated_tokens,
    };

    // 5. Decide LLM
    let (commit_message, used_llm) = match smart_mode {
        SmartMode::Never => (generate_fallback_message(&stats), false),

        SmartMode::Smart => {
            if stats.is_significant {
                let max_diff = if provider.requires_api_key() { 16_000 } else { 8_000 };
                let diff_text = if diff_content.len() > max_diff {
                    format!("(Diff truncated)...\n{}", &diff_content[..max_diff])
                } else {
                    diff_content.to_string()
                };
                let prompt = format!("Generate a commit message for these changes:\n\n{}", diff_text);
                match call_llm(provider, base_url, model, api_key, &prompt).await {
                    Ok(msg) => (msg, true),
                    Err(_) => (generate_fallback_message(&stats), false),
                }
            } else {
                (generate_fallback_message(&stats), false)
            }
        }

        SmartMode::Always => {
            let max_diff = if provider.requires_api_key() { 16_000 } else { 8_000 };
            let diff_text = if diff_content.len() > max_diff {
                format!("(Diff truncated)...\n{}", &diff_content[..max_diff])
            } else {
                diff_content.to_string()
            };
            let prompt = format!("Generate a commit message for these changes:\n\n{}", diff_text);
            let msg = call_llm(provider, base_url, model, api_key, &prompt).await?;
            (msg, true)
        }
    };

    let mut clean_message = commit_message
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();

    // Apply prefix
    if !commit_prefix.is_empty() {
        clean_message = format!("{} {}", commit_prefix.trim(), clean_message);
    }

    // Dry run: return early without committing
    if dry_run {
        return Ok(CommitResult {
            message: format!("[DRY RUN] {}", clean_message),
            used_llm,
            diff_stats: Some(stats_public),
        });
    }

    // 6. Commit
    Command::new("git")
        .args(["commit", "-m", &clean_message])
        .current_dir(path)
        .status()
        .map_err(|e| format!("Git commit error: {}", e))?;

    // 7. Push (optional)
    if push_enabled {
        // Parse "origin/main" → remote="origin", branch="main"
        let parts: Vec<&str> = push_branch.splitn(2, '/').collect();
        let (remote, branch) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            ("origin", push_branch)
        };
        Command::new("git")
            .args(["push", remote, branch])
            .current_dir(path)
            .status()
            .map_err(|e| format!("Git push error: {}", e))?;
    }

    Ok(CommitResult {
        message: clean_message,
        used_llm,
        diff_stats: Some(stats_public),
    })
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

fn mask_api_key(key: &str) -> String {
    if key.len() <= 4 {
        return "****".to_string();
    }
    let suffix = &key[key.len() - 4..];
    format!("sk-...{}", suffix)
}

// ---------- TAURI COMMANDS ----------

#[tauri::command]
async fn run_commit(path: String, state: State<'_, AppState>) -> Result<CommitResult> {
    let (provider, base_url, model, api_key, smart_mode, threshold,
        push_enabled, push_branch, commit_prefix, cooldown, last_commit_time) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        (
            c.provider.clone(),
            c.llm_base_url.clone(),
            c.llm_model_name.clone(),
            c.llm_api_key.clone(),
            c.smart_mode.clone(),
            c.smart_threshold_lines,
            c.push_enabled,
            c.push_branch.clone(),
            c.commit_prefix.clone(),
            c.cooldown_minutes,
            c.commit_history.last().map(|h| h.timestamp).unwrap_or(0),
        )
    };
    let result = run_commit_internal(
        &path, &provider, &base_url, &model, &api_key, &smart_mode, threshold,
        push_enabled, &push_branch, &commit_prefix, cooldown, last_commit_time, false,
    ).await?;

    // Save to history
    if result.message != "No changes to commit" && !result.message.starts_with("Cooldown") {
        let stats = result.diff_stats.clone().unwrap_or(DiffStatsPublic {
            files_changed: 0, insertions: 0, deletions: 0, estimated_tokens: 0,
        });
        let entry = CommitHistoryEntry {
            timestamp: now_unix(),
            repo_path: path,
            message: result.message.clone(),
            used_llm: result.used_llm,
            files_changed: stats.files_changed,
            insertions: stats.insertions,
            deletions: stats.deletions,
            estimated_tokens: stats.estimated_tokens,
        };
        let mut c = state.config.lock().map_err(|e| e.to_string())?;
        c.commit_history.push(entry);
        // Persist
        let config_path = get_config_path()?;
        if let Ok(json) = serde_json::to_string_pretty(&*c) {
            let _ = fs::write(&config_path, json);
        }
    }

    Ok(result)
}

#[tauri::command]
async fn dry_run_commit(path: String, state: State<'_, AppState>) -> Result<CommitResult> {
    let (provider, base_url, model, api_key, smart_mode, threshold, commit_prefix) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        (
            c.provider.clone(),
            c.llm_base_url.clone(),
            c.llm_model_name.clone(),
            c.llm_api_key.clone(),
            c.smart_mode.clone(),
            c.smart_threshold_lines,
            c.commit_prefix.clone(),
        )
    };
    run_commit_internal(
        &path, &provider, &base_url, &model, &api_key, &smart_mode, threshold,
        false, "origin/main", &commit_prefix, 0, 0, true,
    ).await
}

#[tauri::command]
async fn save_config(config: AppConfig, state: State<'_, AppState>) -> Result<()> {
    let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
    *app_config = config.clone();
    let config_path = get_config_path()?;
    let json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Serialize error: {}", e))?;
    fs::write(&config_path, json)
        .map_err(|e| format!("Write error: {}", e))?;
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
        let s = fs::read_to_string(config_path)
            .map_err(|e| format!("Read error: {}", e))?;
        let config: AppConfig = serde_json::from_str(&s).unwrap_or_default();
        let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
        *app_config = config.clone();
        Ok(config)
    } else {
        Ok(AppConfig::default())
    }
}

#[tauri::command]
async fn get_provider_defaults(provider: LlmProvider) -> Result<(String, String)> {
    Ok((provider.default_base_url().to_string(), provider.default_model().to_string()))
}

#[tauri::command]
async fn get_masked_api_key(state: State<'_, AppState>) -> Result<String> {
    let c = state.config.lock().map_err(|e| e.to_string())?;
    Ok(mask_api_key(&c.llm_api_key))
}

#[tauri::command]
async fn validate_repo_path(path: String) -> Result<bool> {
    let git_dir = std::path::Path::new(&path).join(".git");
    Ok(git_dir.exists() && git_dir.is_dir())
}

#[tauri::command]
async fn get_current_branch(path: String) -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(&path)
        .output()
        .map_err(|e| format!("Git branch error: {}", e))?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[tauri::command]
async fn list_remote_branches(path: String) -> Result<Vec<String>> {
    let output = Command::new("git")
        .args(["branch", "-r"])
        .current_dir(&path)
        .output()
        .map_err(|e| format!("Git branch list error: {}", e))?;
    let branches: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.contains("HEAD"))
        .collect();
    Ok(branches)
}

#[tauri::command]
async fn get_diff_preview(path: String, state: State<'_, AppState>) -> Result<DiffStatsPublic> {
    let threshold = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        c.smart_threshold_lines
    };
    let diff_output = Command::new("git")
        .args(["diff"])
        .current_dir(&path)
        .output()
        .map_err(|e| format!("Git diff error: {}", e))?;
    let diff_content = String::from_utf8_lossy(&diff_output.stdout);
    let stats = analyze_diff(&diff_content, threshold);
    Ok(DiffStatsPublic {
        files_changed: stats.files_changed,
        insertions: stats.insertions,
        deletions: stats.deletions,
        estimated_tokens: stats.estimated_tokens,
    })
}

#[tauri::command]
async fn get_commit_history(state: State<'_, AppState>) -> Result<Vec<CommitHistoryEntry>> {
    let c = state.config.lock().map_err(|e| e.to_string())?;
    let mut history = c.commit_history.clone();
    history.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(history)
}

#[tauri::command]
async fn export_history_csv(state: State<'_, AppState>) -> Result<String> {
    let c = state.config.lock().map_err(|e| e.to_string())?;
    let mut csv = "timestamp,repo,message,used_llm,files_changed,insertions,deletions,est_tokens\n".to_string();
    for entry in &c.commit_history {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            entry.timestamp,
            entry.repo_path.replace(',', ";"),
            entry.message.replace(',', ";").replace('\n', " "),
            entry.used_llm,
            entry.files_changed,
            entry.insertions,
            entry.deletions,
            entry.estimated_tokens,
        ));
    }
    Ok(csv)
}

#[tauri::command]
async fn clear_commit_history(state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.commit_history.clear();
    let config_path = get_config_path()?;
    if let Ok(json) = serde_json::to_string_pretty(&*c) {
        let _ = fs::write(&config_path, json);
    }
    Ok(())
}

// Multi-repo commands
#[tauri::command]
async fn add_repo(repo: RepoEntry, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.repos.push(repo);
    let config_path = get_config_path()?;
    if let Ok(json) = serde_json::to_string_pretty(&*c) {
        let _ = fs::write(&config_path, json);
    }
    Ok(())
}

#[tauri::command]
async fn remove_repo(id: String, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.repos.retain(|r| r.id != id);
    let config_path = get_config_path()?;
    if let Ok(json) = serde_json::to_string_pretty(&*c) {
        let _ = fs::write(&config_path, json);
    }
    Ok(())
}

#[tauri::command]
async fn update_repo(repo: RepoEntry, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    if let Some(r) = c.repos.iter_mut().find(|r| r.id == repo.id) {
        *r = repo;
    }
    let config_path = get_config_path()?;
    if let Ok(json) = serde_json::to_string_pretty(&*c) {
        let _ = fs::write(&config_path, json);
    }
    Ok(())
}

#[tauri::command]
async fn get_repos(state: State<'_, AppState>) -> Result<Vec<RepoEntry>> {
    let c = state.config.lock().map_err(|e| e.to_string())?;
    Ok(c.repos.clone())
}

#[tauri::command]
async fn start_auto_commit(state: State<'_, AppState>, app_handle: AppHandle) -> Result<()> {
    let (interval_minutes, repo_path, provider, base_url, model, api_key,
        smart_mode, threshold, push_enabled, push_branch, commit_prefix, cooldown) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        (
            c.interval_minutes,
            c.repo_path.clone(),
            c.provider.clone(),
            c.llm_base_url.clone(),
            c.llm_model_name.clone(),
            c.llm_api_key.clone(),
            c.smart_mode.clone(),
            c.smart_threshold_lines,
            c.push_enabled,
            c.push_branch.clone(),
            c.commit_prefix.clone(),
            c.cooldown_minutes,
        )
    };

    if interval_minutes == 0 { return Err("Interval must be at least 1 minute.".into()); }

    let mut running = state.timer_running.lock().map_err(|e| e.to_string())?;
    if *running { return Err("Timer is already running".into()); }
    *running = true;
    drop(running);

    let timer_arc = Arc::clone(&state.timer_running);
    let config_arc = Arc::clone(&state.config);

    tauri::async_runtime::spawn(async move {
        let mut tick = interval(Duration::from_secs(interval_minutes * 60));
        loop {
            tick.tick().await;
            if !*timer_arc.lock().unwrap() { break; }

            let last_commit_time = config_arc.lock().unwrap()
                .commit_history.last().map(|h| h.timestamp).unwrap_or(0);

            match run_commit_internal(
                &repo_path, &provider, &base_url, &model, &api_key,
                &smart_mode, threshold, push_enabled, &push_branch,
                &commit_prefix, cooldown, last_commit_time, false,
            ).await {
                Ok(r) => {
                    if r.message != "No changes to commit" && !r.message.starts_with("Cooldown") {
                        let stats = r.diff_stats.clone().unwrap_or(DiffStatsPublic {
                            files_changed: 0, insertions: 0, deletions: 0, estimated_tokens: 0,
                        });
                        let entry = CommitHistoryEntry {
                            timestamp: now_unix(),
                            repo_path: repo_path.clone(),
                            message: r.message.clone(),
                            used_llm: r.used_llm,
                            files_changed: stats.files_changed,
                            insertions: stats.insertions,
                            deletions: stats.deletions,
                            estimated_tokens: stats.estimated_tokens,
                        };
                        {
                            let mut c = config_arc.lock().unwrap();
                            c.commit_history.push(entry);
                            if let Ok(json) = serde_json::to_string_pretty(&*c) {
                                if let Ok(path) = get_config_path() {
                                    let _ = fs::write(path, json);
                                }
                            }
                        }
                        let _ = app_handle.emit("commit-status", r);
                    }
                }
                Err(e) => { let _ = app_handle.emit("commit-error", e); }
            }
        }
    });
    Ok(())
}

#[tauri::command]
async fn stop_auto_commit(state: State<'_, AppState>) -> Result<()> {
    let mut running = state.timer_running.lock().map_err(|e| e.to_string())?;
    *running = false;
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
    call_llm(&provider, &base_url, &model, &api_key, "Reply with exactly the word 'Connected'.")
        .await
        .map(|_| "Connection successful!".to_string())
}

// ---------- MAIN ----------

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            run_commit,
            dry_run_commit,
            save_config,
            get_config,
            load_config_from_file,
            get_provider_defaults,
            get_masked_api_key,
            validate_repo_path,
            get_current_branch,
            list_remote_branches,
            get_diff_preview,
            get_commit_history,
            export_history_csv,
            clear_commit_history,
            add_repo,
            remove_repo,
            update_repo,
            get_repos,
            start_auto_commit,
            stop_auto_commit,
            select_directory,
            test_connection,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}