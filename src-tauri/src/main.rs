use tauri::{AppHandle, Emitter, State, Manager};
use tauri_plugin_dialog::DialogExt;
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
    LmStudio, Ollama, LocalAi, Custom,
    OpenAi, Groq, Gemini, Anthropic, Mistral, Together, OpenRouter,
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
        matches!(self,
            LlmProvider::OpenAi | LlmProvider::Groq | LlmProvider::Gemini |
            LlmProvider::Anthropic | LlmProvider::Mistral |
            LlmProvider::Together | LlmProvider::OpenRouter
        )
    }

    fn is_anthropic(&self) -> bool { *self == LlmProvider::Anthropic }
}

impl Default for LlmProvider {
    fn default() -> Self { LlmProvider::LmStudio }
}

// ---------- SMART MODE ----------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SmartMode { Always, Smart, Never }

impl Default for SmartMode {
    fn default() -> Self { SmartMode::Smart }
}

// ---------- DIFF STATS ----------

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
    let total = (insertions + deletions) as u64;
    DiffStats {
        files_changed, insertions, deletions,
        is_significant: total >= threshold_lines || files_changed >= 3,
        estimated_tokens: diff.len() / 4,
    }
}

fn generate_fallback_message(stats: &DiffStats) -> String {
    if stats.files_changed == 0 { return "chore: minor update".to_string(); }
    if stats.deletions > stats.insertions * 2 {
        return format!("refactor: remove code across {} file(s)", stats.files_changed);
    }
    if stats.insertions > 0 && stats.deletions == 0 {
        return format!("feat: add new code in {} file(s)", stats.files_changed);
    }
    format!("chore: update {} file(s) (+{} -{} lines)", stats.files_changed, stats.insertions, stats.deletions)
}

// ---------- MULTI-REPO ----------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepoEntry {
    pub id: String,
    pub path: String,
    pub interval_minutes: u64,
    pub enabled: bool,
    pub push_enabled: bool,
    pub push_remote: String,
    pub push_branch: String,
    pub commit_prefix: String,
    pub last_commit_time: u64,
    pub cooldown_minutes: u64,
}

// ---------- HISTORY ----------

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
    push_enabled: bool,
    push_remote: String,
    push_branch: String,
    commit_prefix: String,
    cooldown_minutes: u64,
    // Timestamp del último commit exitoso (para cooldown del timer)
    human_in_the_loop: bool,
    last_successful_commit: u64,
    repos: Vec<RepoEntry>,
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
            push_remote: "origin".to_string(),
            push_branch: "main".to_string(),
            commit_prefix: String::new(),
            cooldown_minutes: 5,
            human_in_the_loop: true,
            last_successful_commit: 0,
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
struct Message { role: String, content: String }

#[derive(Deserialize)]
struct ChatCompletionResponse { choices: Vec<Choice> }

#[derive(Deserialize)]
struct Choice { message: Message }

#[derive(Serialize)]
struct AnthropicRequest {
    model: String, messages: Vec<Message>,
    max_tokens: u32, system: String,
}

#[derive(Deserialize)]
struct AnthropicResponse { content: Vec<AnthropicContent> }

#[derive(Deserialize)]
struct AnthropicContent { text: String }

const SYSTEM_PROMPT: &str = r#"
You are an expert Git commit message generator.
Analyze the provided git diff and generate a concise, conventional commit message.
Format: <type>(<optional scope>): <description>
Types: feat, fix, docs, style, refactor, test, chore.
Example: feat: add multi-provider LLM support
IMPORTANT: Respond ONLY with the commit message. No quotes, no explanations, no markdown.
"#;

// ---------- LLM CALL ----------

use once_cell::sync::Lazy;
static HTTP_CLIENT: Lazy<Client> = Lazy::new(Client::new);

async fn call_llm(
    provider: &LlmProvider, base_url: &str,
    model: &str, api_key: &str, user_content: &str,
) -> Result<String> {
    if provider.is_anthropic() {
        let endpoint = format!("{}/messages", base_url.trim_end_matches('/'));
        let body = AnthropicRequest {
            model: model.to_string(), system: SYSTEM_PROMPT.to_string(),
            max_tokens: 256,
            messages: vec![Message { role: "user".to_string(), content: user_content.to_string() }],
        };
        let resp = HTTP_CLIENT.post(&endpoint)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body).send().await
            .map_err(|e| format!("Anthropic connection failed: {}", e))?;
        if !resp.status().is_success() {
            return Err(format!("Anthropic error: {}", resp.text().await.unwrap_or_default()));
        }
        let parsed: AnthropicResponse = resp.json().await
            .map_err(|e| format!("Parse error: {}", e))?;
        return parsed.content.into_iter().next()
            .map(|c| c.text)
            .ok_or_else(|| "Anthropic returned no content".to_string());
    }

    let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = ChatCompletionRequest {
        model: model.to_string(), temperature: 0.3, max_tokens: Some(256),
        messages: vec![
            Message { role: "system".to_string(), content: SYSTEM_PROMPT.to_string() },
            Message { role: "user".to_string(), content: user_content.to_string() },
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

    let resp = req.send().await
        .map_err(|e| format!("Connection to {} failed: {}", base_url, e))?;
    if !resp.status().is_success() {
        return Err(format!("LLM API error: {}", resp.text().await.unwrap_or_default()));
    }
    let parsed: ChatCompletionResponse = resp.json().await
        .map_err(|e| format!("Parse error: {}", e))?;
    parsed.choices.into_iter().next()
        .map(|c| c.message.content)
        .ok_or_else(|| "LLM returned no content".to_string())
}

// ---------- PUBLIC TYPES ----------

#[derive(Serialize, Deserialize, Clone)]
pub struct DiffStatsPublic {
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub estimated_tokens: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PendingCommit {
    pub message: String,
    pub used_llm: bool,
    pub diff_stats: DiffStatsPublic,
    pub diff_preview: String,      // primeras líneas del diff para mostrar al usuario
    pub files_changed_list: Vec<String>, // lista de ficheros afectados
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CommitResult {
    pub message: String,
    pub used_llm: bool,
    pub diff_stats: Option<DiffStatsPublic>,
    pub pending_approval: Option<PendingCommit>,
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ---------- CORE COMMIT LOGIC ----------

async fn llm_commit_message(
    provider: &LlmProvider, base_url: &str, model: &str,
    api_key: &str, diff_content: &str, stats: &DiffStats,
) -> (String, bool) {
    let max_diff = if provider.requires_api_key() { 16_000 } else { 8_000 };
    let diff_text = if diff_content.len() > max_diff {
        format!("(Diff truncated)...\n{}", &diff_content[..max_diff])
    } else { diff_content.to_string() };
    let prompt = format!("Generate a commit message for these changes:\n\n{}", diff_text);
    match call_llm(provider, base_url, model, api_key, &prompt).await {
        Ok(msg) => (msg, true),
        Err(_) => (generate_fallback_message(stats), false),
    }
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
    push_remote: &str,
    push_branch: &str,
    commit_prefix: &str,
    cooldown_minutes: u64,
    last_commit_time: u64,
    dry_run: bool,
    human_in_the_loop: bool,
) -> Result<CommitResult> {
    // Cooldown check
    if !dry_run && cooldown_minutes > 0 {
        let elapsed = now_unix().saturating_sub(last_commit_time);
        if elapsed < cooldown_minutes * 60 {
            return Ok(CommitResult {
                message: format!("Cooldown active: {}s remaining", cooldown_minutes * 60 - elapsed),
                used_llm: false,
                diff_stats: None,
                pending_approval: None,
            });
        }
    }

    // 1. Comprobar cambios
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(path)
        .output()
        .map_err(|e| format!("Git status error: {}", e))?;

    if String::from_utf8_lossy(&status.stdout).trim().is_empty() {
        return Ok(CommitResult {
            message: "No changes to commit".to_string(),
            used_llm: false,
            diff_stats: None,
            pending_approval: None,
        });
    }

    // 2. Stage
    if !dry_run {
        Command::new("git").args(["add", "."]).current_dir(path)
            .status().map_err(|e| format!("Git add error: {}", e))?;
    }

    // 3. Diff
    let diff_args = if dry_run { vec!["diff"] } else { vec!["diff", "--cached"] };
    let diff_out = Command::new("git").args(&diff_args).current_dir(path)
        .output().map_err(|e| format!("Git diff error: {}", e))?;
    let diff_content = String::from_utf8_lossy(&diff_out.stdout);

    // 4. Analyze
    let stats = analyze_diff(&diff_content, smart_threshold_lines);
    let stats_public = DiffStatsPublic {
        files_changed: stats.files_changed,
        insertions: stats.insertions,
        deletions: stats.deletions,
        estimated_tokens: stats.estimated_tokens,
    };

    // 5. LLM decision
    let (commit_message, used_llm) = match smart_mode {
        SmartMode::Never => (generate_fallback_message(&stats), false),

        SmartMode::Smart => {
            if stats.is_significant {
                llm_commit_message(provider, base_url, model, api_key, &diff_content, &stats).await
            } else {
                (generate_fallback_message(&stats), false)
            }
        }

        SmartMode::Always => {
            llm_commit_message(provider, base_url, model, api_key, &diff_content, &stats).await
        }
    };

    let mut clean_message = commit_message.trim_matches('"').trim_matches('\'').trim().to_string();
    if !commit_prefix.is_empty() {
        clean_message = format!("{} {}", commit_prefix.trim(), clean_message);
    }

    // Dry run → salir antes de commitear
    if dry_run {
        return Ok(CommitResult {
            message: format!("[DRY RUN] {}", clean_message),
            used_llm,
            diff_stats: Some(stats_public),
            pending_approval: None,
        });
    }

    let files_changed_list: Vec<String> = {
        Command::new("git")
            .args(["diff", "--cached", "--name-only"])
            .current_dir(path)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect())
            .unwrap_or_default()
    };

    // Primeras 60 líneas del diff como preview
    let diff_preview: String = diff_content
        .lines()
        .take(60)
        .collect::<Vec<_>>()
        .join("\n");

    // Human in the loop — devolver pending antes de commitear
    if human_in_the_loop && !dry_run {
        return Ok(CommitResult {
            message: clean_message.clone(),
            used_llm,
            diff_stats: Some(stats_public.clone()),
            pending_approval: Some(PendingCommit {
                message: clean_message,
                used_llm,
                diff_stats: stats_public,
                diff_preview,
                files_changed_list,
            }),
        });
    }

    // 6. Commit
    Command::new("git").args(["commit", "-m", &clean_message]).current_dir(path)
        .status().map_err(|e| format!("Git commit error: {}", e))?;

    // 7. Push opcional
    if push_enabled {
        Command::new("git").args(["push", push_remote, push_branch]).current_dir(path)
            .status().map_err(|e| format!("Git push error: {}", e))?;
    }

    Ok(CommitResult { message: clean_message, used_llm, diff_stats: Some(stats_public), pending_approval: None })}

// ---------- HELPERS ----------

fn get_config_path() -> Result<PathBuf> {
    let mut path = dirs::config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    path.push("auto-commit-app");
    fs::create_dir_all(&path).map_err(|e| format!("Create dir error: {}", e))?;
    path.push("config.json");
    Ok(path)
}

fn mask_api_key(key: &str) -> String {
    if key.len() <= 4 { return "****".to_string(); }
    format!("sk-...{}", &key[key.len()-4..])
}

fn persist_config(config: &AppConfig) {
    if let Ok(path) = get_config_path() {
        if let Ok(json) = serde_json::to_string_pretty(config) {
            let _ = fs::write(path, json);
        }
    }
}

fn push_history(config: &mut AppConfig, entry: CommitHistoryEntry) {
    config.commit_history.push(entry);
    config.last_successful_commit = now_unix();
    persist_config(config);
}

// ---------- TAURI COMMANDS ----------

#[tauri::command]
async fn run_commit(path: String, state: State<'_, AppState>) -> Result<CommitResult> {
    // let (provider, base_url, model, api_key, smart_mode, threshold,
    //     push_enabled, push_branch, commit_prefix, cooldown, last_commit) = {
    //     let c = state.config.lock().map_err(|e| e.to_string())?;
    //     (c.provider.clone(), c.llm_base_url.clone(), c.llm_model_name.clone(),
    //      c.llm_api_key.clone(), c.smart_mode.clone(), c.smart_threshold_lines,
    //      c.push_enabled, c.push_branch.clone(), c.commit_prefix.clone(),
    //      c.cooldown_minutes, c.last_successful_commit)
    // };

    let (provider, base_url, model, api_key, smart_mode, threshold,
        push_enabled, push_remote, push_branch, commit_prefix, cooldown, last_commit, hitl) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        (c.provider.clone(), c.llm_base_url.clone(), c.llm_model_name.clone(),
         c.llm_api_key.clone(), c.smart_mode.clone(), c.smart_threshold_lines,
         c.push_enabled, c.push_remote.clone(), c.push_branch.clone(), c.commit_prefix.clone(),
         c.cooldown_minutes, c.last_successful_commit, c.human_in_the_loop)
    };


    let result = run_commit_internal(
        &path, &provider, &base_url, &model, &api_key,
        &smart_mode, threshold, push_enabled, &push_remote, &push_branch,
        &commit_prefix, cooldown, last_commit, false, hitl,
    ).await?;

    if result.pending_approval.is_none()
        && result.message != "No changes to commit"
        && !result.message.starts_with("Cooldown") {
        let stats = result.diff_stats.clone().unwrap_or(DiffStatsPublic {
            files_changed: 0, insertions: 0, deletions: 0, estimated_tokens: 0,
        });
        let entry = CommitHistoryEntry {
            timestamp: now_unix(), repo_path: path,
            message: result.message.clone(), used_llm: result.used_llm,
            files_changed: stats.files_changed, insertions: stats.insertions,
            deletions: stats.deletions, estimated_tokens: stats.estimated_tokens,
        };
        let mut c = state.config.lock().map_err(|e| e.to_string())?;
        push_history(&mut c, entry);
    }
    Ok(result)
}

// NUEVO — comando para ejecutar el commit tras aprobación
#[tauri::command]
async fn confirm_commit(
    path: String,
    message: String,
    push_enabled: bool,
    state: State<'_, AppState>,
) -> Result<CommitResult> {
    let (push_remote, push_branch) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        (c.push_remote.clone(), c.push_branch.clone())
    };

    // Stage
    Command::new("git").args(["add", "."]).current_dir(&path)
        .status().map_err(|e| format!("Git add error: {}", e))?;

    // Commit con el mensaje (posiblemente editado por el usuario)
    Command::new("git").args(["commit", "-m", &message]).current_dir(&path)
        .status().map_err(|e| format!("Git commit error: {}", e))?;

    // Push opcional
    if push_enabled {
        Command::new("git").args(["push", &push_remote, &push_branch]).current_dir(&path)
            .status().map_err(|e| format!("Git push error: {}", e))?;
    }

    let entry = CommitHistoryEntry {
        timestamp: now_unix(), repo_path: path,
        message: message.clone(), used_llm: false,
        files_changed: 0, insertions: 0, deletions: 0, estimated_tokens: 0,
    };
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    push_history(&mut c, entry);

    Ok(CommitResult {
        message,
        used_llm: false,
        diff_stats: None,
        pending_approval: None,
    })
}

#[tauri::command]
async fn dry_run_commit(path: String, state: State<'_, AppState>) -> Result<CommitResult> {
    let (provider, base_url, model, api_key, smart_mode, threshold, commit_prefix) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        (c.provider.clone(), c.llm_base_url.clone(), c.llm_model_name.clone(),
         c.llm_api_key.clone(), c.smart_mode.clone(), c.smart_threshold_lines,
         c.commit_prefix.clone())
    };
    run_commit_internal(
        &path, &provider, &base_url, &model, &api_key,
        &smart_mode, threshold, false, "origin", "main",
        &commit_prefix, 0, 0, true, false
    ).await
}

#[tauri::command]
async fn save_config(config: AppConfig, state: State<'_, AppState>) -> Result<()> {
    let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
    // Preservar historial y último commit al guardar config desde UI
    let history = app_config.commit_history.clone();
    let last = app_config.last_successful_commit;
    *app_config = config;
    app_config.commit_history = history;
    app_config.last_successful_commit = last;
    persist_config(&app_config);
    Ok(())
}

#[tauri::command]
async fn get_config(state: State<'_, AppState>) -> Result<AppConfig> {
    Ok(state.config.lock().map_err(|e| e.to_string())?.clone())
}

#[tauri::command]
async fn load_config_from_file(state: State<'_, AppState>) -> Result<AppConfig> {
    let config_path = get_config_path()?;
    if config_path.exists() {
        let s = fs::read_to_string(config_path).map_err(|e| format!("Read error: {}", e))?;
        let config: AppConfig = serde_json::from_str(&s).unwrap_or_default();
        *state.config.lock().map_err(|e| e.to_string())? = config.clone();
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
    Ok(mask_api_key(&state.config.lock().map_err(|e| e.to_string())?.llm_api_key))
}

#[tauri::command]
async fn validate_repo_path(path: String) -> Result<bool> {
    Ok(std::path::Path::new(&path).join(".git").is_dir())
}

#[tauri::command]
async fn get_current_branch(path: String) -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(&path)
        .output()
        .map_err(|e| format!("Git branch error: {}", e))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[tauri::command]
async fn list_remote_branches(path: String) -> Result<Vec<String>> {
    let out = Command::new("git").args(["branch", "-r"]).current_dir(&path)
        .output().map_err(|e| format!("Git remote branch error: {}", e))?;
    Ok(String::from_utf8_lossy(&out.stdout).lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.contains("HEAD"))
        .collect())
}

#[tauri::command]
async fn get_diff_preview(path: String, state: State<'_, AppState>) -> Result<DiffStatsPublic> {
    let threshold = state.config.lock().map_err(|e| e.to_string())?.smart_threshold_lines;
    let out = Command::new("git").args(["diff"]).current_dir(&path)
        .output().map_err(|e| format!("Git diff error: {}", e))?;
    let diff = String::from_utf8_lossy(&out.stdout);
    let stats = analyze_diff(&diff, threshold);
    Ok(DiffStatsPublic {
        files_changed: stats.files_changed,
        insertions: stats.insertions,
        deletions: stats.deletions,
        estimated_tokens: stats.estimated_tokens,
    })
}

#[tauri::command]
async fn get_commit_history(state: State<'_, AppState>) -> Result<Vec<CommitHistoryEntry>> {
    let mut h = state.config.lock().map_err(|e| e.to_string())?.commit_history.clone();
    h.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(h)
}

#[tauri::command]
async fn export_history_csv(state: State<'_, AppState>) -> Result<String> {
    let c = state.config.lock().map_err(|e| e.to_string())?;
    let mut csv = "timestamp,repo,message,used_llm,files_changed,insertions,deletions,est_tokens\n".to_string();
    for e in &c.commit_history {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            e.timestamp,
            e.repo_path.replace(',', ";"),
            format!("\"{}\"", e.message.replace('"', "\"\"").replace('\n', " ")),
            e.used_llm, e.files_changed, e.insertions, e.deletions, e.estimated_tokens,
        ));
    }
    Ok(csv)
}

#[tauri::command]
async fn clear_commit_history(state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.commit_history.clear();
    persist_config(&c);
    Ok(())
}

#[tauri::command]
async fn add_repo(repo: RepoEntry, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.repos.push(repo);
    persist_config(&c);
    Ok(())
}

#[tauri::command]
async fn remove_repo(id: String, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.repos.retain(|r| r.id != id);
    persist_config(&c);
    Ok(())
}

#[tauri::command]
async fn update_repo(repo: RepoEntry, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    if let Some(r) = c.repos.iter_mut().find(|r| r.id == repo.id) { *r = repo; }
    persist_config(&c);
    Ok(())
}

#[tauri::command]
async fn get_repos(state: State<'_, AppState>) -> Result<Vec<RepoEntry>> {
    Ok(state.config.lock().map_err(|e| e.to_string())?.repos.clone())
}

// ─── TIMER — FIX: re-lee config completa en cada tick ─────────────────────

#[tauri::command]
async fn start_auto_commit(state: State<'_, AppState>, app_handle: AppHandle) -> Result<()> {
    let interval_minutes = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        if c.interval_minutes == 0 { return Err("Interval must be at least 1 minute.".into()); }
        c.interval_minutes
    };

    {
        let mut running = state.timer_running.lock().map_err(|e| e.to_string())?;
        if *running { return Err("Timer is already running".into()); }
        *running = true;
    }

    let timer_arc  = Arc::clone(&state.timer_running);
    let config_arc = Arc::clone(&state.config);

    tauri::async_runtime::spawn(async move {
        let mut tick = interval(Duration::from_secs(interval_minutes * 60));
        tick.tick().await; // salta el tick inmediato al arrancar

        loop {
            tick.tick().await;

            let Ok(running) = timer_arc.lock() else { break; };
            if !*running { break; }

            // Re-leer config completa en cada tick para recoger cambios del usuario
            let (repo_path, provider, base_url, model, api_key,
                smart_mode, threshold, push_enabled, push_remote, push_branch,
                commit_prefix, cooldown, last_commit) = {
                let Ok(c) = config_arc.lock() else { continue; };
                (
                    c.repo_path.clone(),
                    c.provider.clone(),
                    c.llm_base_url.clone(),
                    c.llm_model_name.clone(),
                    c.llm_api_key.clone(),
                    c.smart_mode.clone(),
                    c.smart_threshold_lines,
                    c.push_enabled,
                    c.push_remote.clone(),
                    c.push_branch.clone(),
                    c.commit_prefix.clone(),
                    c.cooldown_minutes,
                    c.last_successful_commit,
                )
            };

            if repo_path.is_empty() { continue; }

            match run_commit_internal(
                &repo_path, &provider, &base_url, &model, &api_key,
                &smart_mode, threshold, push_enabled, &push_remote, &push_branch,
                &commit_prefix, cooldown, last_commit, false, true
            ).await {
                Ok(r) => {
                    let skip = r.pending_approval.is_some()
                        || r.message == "No changes to commit"
                        || r.message.starts_with("Cooldown");

                    if !skip {
                        let stats = r.diff_stats.clone().unwrap_or(DiffStatsPublic {
                            files_changed: 0, insertions: 0,
                            deletions: 0, estimated_tokens: 0,
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
                        // Actualizar state y persistir
                        let Ok(mut c) = config_arc.lock() else { continue; };
                        push_history(&mut c, entry);
                        drop(c);

                        let _ = app_handle.emit("commit-status", &r);
                    }
                }
                Err(e) => { let _ = app_handle.emit("commit-error", &e); }
            }
        }
    });

    Ok(())
}

#[tauri::command]
async fn stop_auto_commit(state: State<'_, AppState>) -> Result<()> {
    *state.timer_running.lock().map_err(|e| e.to_string())? = false;
    Ok(())
}

// ─── FILE PICKER — usando tauri-plugin-dialog ──────────────────────────────

#[tauri::command]
async fn select_directory(app_handle: AppHandle) -> Result<String> {
    let path = app_handle
        .dialog()
        .file()
        .set_title("Selecciona el repositorio Git")
        .blocking_pick_folder();

    match path {
        Some(p) => Ok(p.to_string().replace('\\', "/")),
        None    => Err("No folder selected".to_string()),
    }
}

#[tauri::command]
async fn test_connection(
    provider: LlmProvider, base_url: String,
    model: String, api_key: String,
) -> Result<String> {
    call_llm(&provider, &base_url, &model, &api_key,
             "Reply with exactly the word 'Connected'.")
        .await
        .map(|_| "Connection successful!".to_string())
}

// ---------- MAIN ----------

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            run_commit,
            dry_run_commit,
            confirm_commit,
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