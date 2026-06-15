use std::fs;
use std::process::Command;
use tauri::State;
use tauri_plugin_dialog::DialogExt;

use crate::config::{
    AppConfig, AppState, CommitHistoryEntry, CommitResult, DiffStatsPublic,
    LlmProvider, RepoEntry, Result, get_config_path, mask_api_key, now_unix,
    persist_config, push_history,
};
use crate::git::{analyze_diff, run_commit_internal};
use crate::llm::call_llm;

#[tauri::command]
pub async fn run_commit(
    path: String,
    state: State<'_, AppState>,
) -> Result<CommitResult> {
    let (provider, base_url, model, api_key, smart_mode, threshold, hitl,
        push_enabled, push_remote, push_branch, commit_prefix, cooldown, last_commit) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        // Buscar el repo concreto por path, si no existe usar globales
        let repo = c.repos.iter().find(|r| r.path == path);
        (
            c.provider.clone(),
            c.llm_base_url.clone(),
            c.llm_model_name.clone(),
            c.llm_api_key.clone(),
            c.smart_mode.clone(),
            c.smart_threshold_lines,
            c.human_in_the_loop,
            repo.map(|r| r.push_enabled).unwrap_or(c.push_enabled),
            repo.map(|r| r.push_remote.clone()).unwrap_or(c.push_remote.clone()),
            repo.map(|r| r.push_branch.clone()).unwrap_or(c.push_branch.clone()),
            repo.map(|r| r.commit_prefix.clone()).unwrap_or(c.commit_prefix.clone()),
            repo.map(|r| r.cooldown_minutes).unwrap_or(c.cooldown_minutes),
            repo.map(|r| r.last_commit_time).unwrap_or(c.last_successful_commit),
        )
    };

    let result = run_commit_internal(
        &path, &provider, &base_url, &model, &api_key,
        &smart_mode, threshold, push_enabled, &push_remote, &push_branch,
        &commit_prefix, cooldown, last_commit, false, hitl,
    ).await?;

    if result.pending_approval.is_none()
        && result.message != "No changes to commit"
        && !result.message.starts_with("Cooldown")
    {
        let stats = result.diff_stats.clone().unwrap_or(DiffStatsPublic {
            files_changed: 0, insertions: 0, deletions: 0, estimated_tokens: 0,
        });
        let entry = CommitHistoryEntry {
            timestamp: now_unix(), repo_path: path.clone(),
            message: result.message.clone(), used_llm: result.used_llm,
            files_changed: stats.files_changed, insertions: stats.insertions,
            deletions: stats.deletions, estimated_tokens: stats.estimated_tokens,
        };
        let mut c = state.config.lock().map_err(|e| e.to_string())?;
        // Actualizar last_commit_time del repo específico
        if let Some(r) = c.repos.iter_mut().find(|r| r.path == path) {
            r.last_commit_time = now_unix();
        }
        push_history(&mut c, entry);
    }
    Ok(result)
}

#[tauri::command]
pub async fn confirm_commit(
    path: String,
    message: String,
    push_enabled: bool,
    state: State<'_, AppState>,
) -> Result<CommitResult> {
    // Leer remote/branch del repo específico
    let (push_remote, push_branch) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        let repo = c.repos.iter().find(|r| r.path == path);
        (
            repo.map(|r| r.push_remote.clone()).unwrap_or(c.push_remote.clone()),
            repo.map(|r| r.push_branch.clone()).unwrap_or(c.push_branch.clone()),
        )
    };

    Command::new("git").args(["add", "."]).current_dir(&path)
        .status().map_err(|e| format!("Git add error: {}", e))?;

    Command::new("git").args(["commit", "-m", &message]).current_dir(&path)
        .status().map_err(|e| format!("Git commit error: {}", e))?;

    if push_enabled {
        Command::new("git")
            .args(["push", &push_remote, &push_branch])
            .current_dir(&path)
            .status()
            .map_err(|e| format!("Git push error: {}", e))?;
    }

    let entry = CommitHistoryEntry {
        timestamp: now_unix(), repo_path: path.clone(),
        message: message.clone(), used_llm: false,
        files_changed: 0, insertions: 0, deletions: 0, estimated_tokens: 0,
    };
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    if let Some(r) = c.repos.iter_mut().find(|r| r.path == path) {
        r.last_commit_time = now_unix();
    }
    push_history(&mut c, entry);

    Ok(CommitResult { message, used_llm: false, diff_stats: None, pending_approval: None })
}

#[tauri::command]
pub async fn dry_run_commit(
    path: String,
    state: State<'_, AppState>,
) -> Result<CommitResult> {
    let (provider, base_url, model, api_key, smart_mode, threshold, commit_prefix) = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        (
            c.provider.clone(), c.llm_base_url.clone(), c.llm_model_name.clone(),
            c.llm_api_key.clone(), c.smart_mode.clone(), c.smart_threshold_lines,
            c.commit_prefix.clone(),
        )
    };
    run_commit_internal(
        &path, &provider, &base_url, &model, &api_key,
        &smart_mode, threshold, false, "origin", "main",
        &commit_prefix, 0, 0, true, false,
    ).await
}

#[tauri::command]
pub async fn save_config(
    config: AppConfig,
    state: State<'_, AppState>,
) -> Result<()> {
    let mut app_config = state.config.lock().map_err(|e| e.to_string())?;
    let history = app_config.commit_history.clone();
    let last    = app_config.last_successful_commit;
    *app_config = config;
    app_config.commit_history = history;
    app_config.last_successful_commit = last;
    persist_config(&app_config);
    Ok(())
}

#[tauri::command]
pub async fn get_config(state: State<'_, AppState>) -> Result<AppConfig> {
    Ok(state.config.lock().map_err(|e| e.to_string())?.clone())
}

#[tauri::command]
pub async fn load_config_from_file(state: State<'_, AppState>) -> Result<AppConfig> {
    let config_path = get_config_path()?;
    if config_path.exists() {
        let s = fs::read_to_string(config_path)
            .map_err(|e| format!("Read error: {}", e))?;
        let config: AppConfig = serde_json::from_str(&s).unwrap_or_default();
        *state.config.lock().map_err(|e| e.to_string())? = config.clone();
        Ok(config)
    } else {
        Ok(AppConfig::default())
    }
}

#[tauri::command]
pub async fn get_provider_defaults(provider: LlmProvider) -> Result<(String, String)> {
    Ok((
        provider.default_base_url().to_string(),
        provider.default_model().to_string(),
    ))
}

#[tauri::command]
pub async fn get_masked_api_key(state: State<'_, AppState>) -> Result<String> {
    Ok(mask_api_key(
        &state.config.lock().map_err(|e| e.to_string())?.llm_api_key,
    ))
}

#[tauri::command]
pub async fn validate_repo_path(path: String) -> Result<bool> {
    Ok(std::path::Path::new(&path).join(".git").is_dir())
}

#[tauri::command]
pub async fn get_current_branch(path: String) -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(&path)
        .output()
        .map_err(|e| format!("Git branch error: {}", e))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[tauri::command]
pub async fn list_remote_branches(path: String) -> Result<Vec<String>> {
    let out = Command::new("git")
        .args(["branch", "-r"])
        .current_dir(&path)
        .output()
        .map_err(|e| format!("Git remote branch error: {}", e))?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.contains("HEAD"))
        .collect())
}

#[tauri::command]
pub async fn get_diff_preview(
    path: String,
    state: State<'_, AppState>,
) -> Result<DiffStatsPublic> {
    let threshold = state
        .config
        .lock()
        .map_err(|e| e.to_string())?
        .smart_threshold_lines;
    let out = Command::new("git")
        .args(["diff"])
        .current_dir(&path)
        .output()
        .map_err(|e| format!("Git diff error: {}", e))?;
    let diff  = String::from_utf8_lossy(&out.stdout);
    let stats = analyze_diff(&diff, threshold);
    Ok(DiffStatsPublic {
        files_changed: stats.files_changed,
        insertions: stats.insertions,
        deletions: stats.deletions,
        estimated_tokens: stats.estimated_tokens,
    })
}

#[tauri::command]
pub async fn get_commit_history(
    state: State<'_, AppState>,
) -> Result<Vec<CommitHistoryEntry>> {
    let mut h = state
        .config
        .lock()
        .map_err(|e| e.to_string())?
        .commit_history
        .clone();
    h.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(h)
}

#[tauri::command]
pub async fn export_history_csv(state: State<'_, AppState>) -> Result<String> {
    let c = state.config.lock().map_err(|e| e.to_string())?;
    let mut csv =
        "timestamp,repo,message,used_llm,files_changed,insertions,deletions,est_tokens\n"
            .to_string();
    for e in &c.commit_history {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            e.timestamp,
            e.repo_path.replace(',', ";"),
            format!("\"{}\"", e.message.replace('"', "\"\"").replace('\n', " ")),
            e.used_llm,
            e.files_changed,
            e.insertions,
            e.deletions,
            e.estimated_tokens,
        ));
    }
    Ok(csv)
}

#[tauri::command]
pub async fn clear_commit_history(state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.commit_history.clear();
    persist_config(&c);
    Ok(())
}

#[tauri::command]
pub async fn add_repo(repo: RepoEntry, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.repos.push(repo);
    persist_config(&c);
    Ok(())
}

#[tauri::command]
pub async fn remove_repo(id: String, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    c.repos.retain(|r| r.id != id);
    persist_config(&c);
    Ok(())
}

#[tauri::command]
pub async fn update_repo(repo: RepoEntry, state: State<'_, AppState>) -> Result<()> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    if let Some(r) = c.repos.iter_mut().find(|r| r.id == repo.id) {
        *r = repo;
    }
    persist_config(&c);
    Ok(())
}

#[tauri::command]
pub async fn get_repos(state: State<'_, AppState>) -> Result<Vec<RepoEntry>> {
    Ok(state.config.lock().map_err(|e| e.to_string())?.repos.clone())
}

#[tauri::command]
pub async fn select_directory(app_handle: tauri::AppHandle) -> Result<String> {
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
pub async fn test_connection(
    provider: LlmProvider,
    base_url: String,
    model: String,
    api_key: String,
) -> Result<String> {
    call_llm(
        &provider, &base_url, &model, &api_key,
        "Reply with exactly the word 'Connected'.",
    )
        .await
        .map(|_| "Connection successful!".to_string())
}