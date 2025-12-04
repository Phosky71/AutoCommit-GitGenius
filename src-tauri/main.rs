use tauri::Manager;
use git2::Repository;
use std::process::Command;
use reqwest::Client;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use tokio::time::{interval, Duration};
use tauri::State;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone)]
struct AppConfig {
    repo_path: String,
    auto_commit_enabled: bool,
    interval_minutes: u64,
    auto_start: bool,
    gemini_api_key: String,
}

#[derive(Default)]
struct AppState {
    config: Arc<Mutex<AppConfig>>,
    timer_running: Arc<Mutex<bool>>,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            repo_path: String::new(),
            auto_commit_enabled: false,
            interval_minutes: 30,
            auto_start: false,
            gemini_api_key: String::new(),
        }
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    #[serde(rename = "systemInstruction")]
    system_instruction: SystemInstruction,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    content: ContentResponse,
}

#[derive(Deserialize)]
struct ContentResponse {
    parts: Vec<PartResponse>,
}

#[derive(Deserialize)]
struct PartResponse {
    text: String,
}

// RAG: System context for commit message generation
const SYSTEM_CONTEXT: &str = r#"You are an expert Git commit message generator specialized in creating professional, concise, and meaningful commit messages following industry best practices.

CONTEXT AND PURPOSE:
- You analyze git diffs to understand code changes
- You generate commit messages following the Conventional Commits specification
- Your primary function is to create clear, actionable commit messages that help developers understand changes at a glance

COMMIT MESSAGE RULES:
1. Format: <type>(<scope>): <subject>
2. Types: feat, fix, docs, style, refactor, test, chore, perf
3. Subject: Imperative mood, lowercase, no period, max 50 characters
4. Be specific and descriptive
5. Focus on WHAT and WHY, not HOW

EXAMPLES:
- feat(auth): add JWT token validation
- fix(api): resolve null pointer in user endpoint
- refactor(database): optimize query performance
- docs(readme): update installation instructions
- style(components): format code with prettier

ANALYSIS APPROACH:
1. Identify modified files and their purpose
2. Determine the type of change (feature, bug fix, etc.)
3. Extract the main impact or goal
4. Formulate a clear, concise message

Always respond with ONLY the commit message, no explanations or additional text."#;

#[tauri::command]
async fn run_commit(path: String, state: State<'_, AppState>) -> Result<String, String> {
    let repo = Repository::open(&path).map_err(|e| e.to_string())?;
    let statuses = repo.statuses(None).map_err(|e| e.to_string())?;
    
    if statuses.is_empty() {
        return Ok("No changes to commit".into());
    }

    // Get API key from config
    let config = state.config.lock().map_err(|e| e.to_string())?;
    let api_key = config.gemini_api_key.clone();
    drop(config);

    if api_key.is_empty() {
        return Err("Gemini API Key not configured. Please add your API key in settings.".into());
    }

    // Stage all changes
    Command::new("git")
        .arg("add")
        .arg(".")
        .current_dir(&path)
        .status()
        .map_err(|e| e.to_string())?;

    // Get diff with context
    let diff = Command::new("git")
        .arg("diff")
        .arg("--cached")
        .arg("--stat")
        .current_dir(&path)
        .output()
        .map_err(|e| e.to_string())?;

    let diff_detailed = Command::new("git")
        .arg("diff")
        .arg("--cached")
        .current_dir(&path)
        .output()
        .map_err(|e| e.to_string())?;

    let diff_stat = String::from_utf8_lossy(&diff.stdout);
    let diff_content = String::from_utf8_lossy(&diff_detailed.stdout);

    // Limit diff size to avoid token limits (max 10000 chars)
    let diff_text = if diff_content.len() > 10000 {
        format!("{}\n\n{}", diff_stat, &diff_content[..10000])
    } else {
        format!("{}\n\n{}", diff_stat, diff_content)
    };

    // Create RAG-enhanced prompt
    let user_prompt = format!(
        "Analyze these git changes and generate a commit message:\n\n{}",
        diff_text
    );

    let client = Client::new();
    
    let request_body = GeminiRequest {
        system_instruction: SystemInstruction {
            parts: vec![Part {
                text: SYSTEM_CONTEXT.to_string(),
            }],
        },
        contents: vec![Content {
            parts: vec![Part {
                text: user_prompt,
            }],
        }],
    };

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={}",
        api_key
    );

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;

    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("Gemini API error: {}", error_text));
    }

    let gemini_response: GeminiResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let commit_message = gemini_response
        .candidates
        .get(0)
        .and_then(|c| c.content.parts.get(0))
        .map(|p| p.text.trim().to_string())
        .ok_or("No commit message generated")?;

    // Clean the message (remove quotes if present)
    let clean_message = commit_message
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();

    // Commit with generated message
    Command::new("git")
        .arg("commit")
        .arg("-m")
        .arg(&clean_message)
        .current_dir(&path)
        .status()
        .map_err(|e| e.to_string())?;

    // Push changes
    Command::new("git")
        .arg("push")
        .current_dir(&path)
        .status()
        .map_err(|e| e.to_string())?;

    Ok(clean_message)
}

#[tauri::command]
async fn save_config(
    config: AppConfig,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
    *app_config = config.clone();
    
    // Persist config to file
    let config_path = get_config_path()?;
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;
    
    fs::write(config_path, config_json)
        .map_err(|e| format!("Failed to save config: {}", e))?;
    
    Ok(())
}

#[tauri::command]
async fn get_config(state: State<'_, AppState>) -> Result<AppConfig, String> {
    let config = state.config.lock().map_err(|e| e.to_string())?;
    Ok(config.clone())
}

#[tauri::command]
async fn load_config_from_file(state: State<'_, AppState>) -> Result<AppConfig, String> {
    let config_path = get_config_path()?;
    
    if config_path.exists() {
        let config_str = fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        
        let config: AppConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config: {}", e))?;
        
        let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
        *app_config = config.clone();
        
        Ok(config)
    } else {
        Ok(AppConfig::default())
    }
}

fn get_config_path() -> Result<PathBuf, String> {
    let mut path = dirs::config_dir()
        .ok_or("Failed to get config directory")?;
    path.push("auto-commit-app");
    fs::create_dir_all(&path)
        .map_err(|e| format!("Failed to create config directory: {}", e))?;
    path.push("config.json");
    Ok(path)
}

#[tauri::command]
async fn start_auto_commit(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let config = state.config.lock().map_err(|e| e.to_string())?;
    let interval_minutes = config.interval_minutes;
    let repo_path = config.repo_path.clone();
    drop(config);

    let mut timer_running = state.timer_running.lock().map_err(|e| e.to_string())?;
    if *timer_running {
        return Err("Timer is already running".into());
    }
    *timer_running = true;
    drop(timer_running);

    let state_clone = state.inner().clone();
    
    tauri::async_runtime::spawn(async move {
        let mut interval_timer = interval(Duration::from_secs(interval_minutes * 60));
        
        loop {
            interval_timer.tick().await;
            
            let timer_running = state_clone.timer_running.lock().unwrap();
            if !*timer_running {
                break;
            }
            drop(timer_running);

            match run_commit(repo_path.clone(), State::from(&state_clone)).await {
                Ok(msg) => {
                    if msg != "No changes to commit" {
                        app_handle.emit_all("commit-status", msg).ok();
                    }
                }
                Err(e) => {
                    app_handle.emit_all("commit-error", e).ok();
                }
            }
        }
    });

    Ok(())
}

#[tauri::command]
async fn stop_auto_commit(state: State<'_, AppState>) -> Result<(), String> {
    let mut timer_running = state.timer_running.lock().map_err(|e| e.to_string())?;
    *timer_running = false;
    Ok(())
}

#[tauri::command]
async fn select_directory() -> Result<String, String> {
    use tauri::api::dialog::blocking::FileDialogBuilder;
    
    let path = FileDialogBuilder::new()
        .pick_folder()
        .ok_or("No folder was selected")?;
    
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
async fn test_api_key(api_key: String) -> Result<String, String> {
    let client = Client::new();
    
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={}",
        api_key
    );

    let test_request = GeminiRequest {
        system_instruction: SystemInstruction {
            parts: vec![Part {
                text: "You are a helpful assistant.".to_string(),
            }],
        },
        contents: vec![Content {
            parts: vec![Part {
                text: "Say 'API Key is valid' if you can read this.".to_string(),
            }],
        }],
    };

    let response = client
        .post(&url)
        .json(&test_request)
        .send()
        .await
        .map_err(|e| format!("Connection error: {}", e))?;

    if response.status().is_success() {
        Ok("API Key is valid!".to_string())
    } else {
        let error_text = response.text().await.unwrap_or_default();
        Err(format!("Invalid API Key: {}", error_text))
    }
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            run_commit,
            save_config,
            get_config,
            load_config_from_file,
            start_auto_commit,
            stop_auto_commit,
            select_directory,
            test_api_key,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
