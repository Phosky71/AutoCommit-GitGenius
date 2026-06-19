use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub type Result<T> = std::result::Result<T, String>;

// ---------- PROVIDERS ----------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    LmStudio, Ollama, LocalAi, Custom,
    OpenAi, Groq, Gemini, Anthropic, Mistral, Together, OpenRouter,
}

impl LlmProvider {
    pub fn default_base_url(&self) -> &'static str {
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
    pub fn default_model(&self) -> &'static str {
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
    pub fn requires_api_key(&self) -> bool {
        matches!(self,
            LlmProvider::OpenAi | LlmProvider::Groq | LlmProvider::Gemini |
            LlmProvider::Anthropic | LlmProvider::Mistral |
            LlmProvider::Together | LlmProvider::OpenRouter
        )
    }
    pub fn is_anthropic(&self) -> bool { *self == LlmProvider::Anthropic }
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

// ---------- MULTI-REPO ----------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepoEntry {
    pub id: String,
    pub path: String,
    pub interval_minutes: u64,
    #[serde(default)]
    pub timer_enabled: bool,
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
pub struct AppConfig {
    pub repo_path: String,
    pub auto_commit_enabled: bool,
    pub interval_minutes: u64,
    pub auto_start: bool,
    pub provider: LlmProvider,
    pub llm_base_url: String,
    pub llm_model_name: String,
    pub llm_api_key: String,
    pub smart_mode: SmartMode,
    pub smart_threshold_lines: u64,
    pub push_enabled: bool,
    pub push_remote: String,
    pub push_branch: String,
    pub commit_prefix: String,
    pub cooldown_minutes: u64,
    pub human_in_the_loop: bool,

    // FIX 1: Se añade #[serde(default)] para que los viejos config.json no fallen al cargar
    #[serde(default)]
    pub git_token: String,

    #[serde(default = "default_theme")]
    pub theme: String,
    pub last_successful_commit: u64,
    pub repos: Vec<RepoEntry>,
    pub commit_history: Vec<CommitHistoryEntry>,
}

fn default_theme() -> String { "dark".to_string() }

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
            git_token: String::new(), // Nuevo campo
            theme: "dark".to_string(),
            last_successful_commit: 0,
            repos: Vec::new(),
            commit_history: Vec::new(),
        }
    }
}

// ---------- STATE ----------

pub struct AppState {
    pub config: Arc<Mutex<AppConfig>>,
    pub timer_running: Arc<Mutex<bool>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            config: Arc::new(Mutex::new(AppConfig::default())),
            timer_running: Arc::new(Mutex::new(false)),
        }
    }
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
    pub diff_preview: String,
    pub files_changed_list: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CommitResult {
    pub message: String,
    pub used_llm: bool,
    pub diff_stats: Option<DiffStatsPublic>,
    pub pending_approval: Option<PendingCommit>,
}

// ---------- HELPERS ----------

pub fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub fn sanitize_timestamp(ts: u64) -> u64 {
    let now = now_unix();
    if ts > now + 60 { 0 } else { ts }
}

pub fn get_config_path() -> Result<PathBuf> {
    let mut path = dirs::config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    path.push("auto-commit-app");
    fs::create_dir_all(&path).map_err(|e| format!("Create dir error: {}", e))?;
    path.push("config.json");
    Ok(path)
}

pub fn mask_api_key(key: &str) -> String {
    let chars_count = key.chars().count();
    if chars_count <= 4 { return "****".to_string(); }

    let last_four: String = key.chars().skip(chars_count - 4).collect();
    format!("sk-...{}", last_four)
}

pub fn persist_config(config: &AppConfig) -> Result<()> {
    let path = get_config_path()?;
    let mut temp_path = path.clone();
    temp_path.set_extension("tmp");

    let json = serde_json::to_string_pretty(config)
        .map_err(|e| format!("JSON serialization error: {}", e))?;

    fs::write(&temp_path, json).map_err(|e| format!("Write error: {}", e))?;
    fs::rename(&temp_path, &path).map_err(|e| format!("Rename error: {}", e))?;

    Ok(())
}

pub fn push_history(config: &mut AppConfig, entry: CommitHistoryEntry) -> Result<()> {
    config.commit_history.push(entry);

    if config.commit_history.len() > 1000 {
        let excess = config.commit_history.len() - 1000;
        config.commit_history.drain(0..excess);
    }

    config.last_successful_commit = now_unix();
    persist_config(config)
}