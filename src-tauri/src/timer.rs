use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::time::{interval, Duration};

use crate::config::{AppState, CommitHistoryEntry, DiffStatsPublic, Result, now_unix, push_history};
use crate::git::run_commit_internal;

#[tauri::command]
pub async fn start_auto_commit(
    state: tauri::State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<()> {
    let interval_minutes = {
        let c = state.config.lock().map_err(|e| e.to_string())?;
        if c.interval_minutes == 0 {
            return Err("Interval must be at least 1 minute.".into());
        }
        c.interval_minutes
    };

    {
        let mut running = state.timer_running.lock().map_err(|e| e.to_string())?;
        if *running {
            return Err("Timer is already running".into());
        }
        *running = true;
    }

    let timer_arc  = Arc::clone(&state.timer_running);
    let config_arc = Arc::clone(&state.config);

    tauri::async_runtime::spawn(async move {
        let mut tick = interval(Duration::from_secs(interval_minutes * 60));
        tick.tick().await; // salta el tick inmediato

        loop {
            tick.tick().await;

            // Leer el flag y soltar el lock en un bloque antes de cualquier await
            let is_running = {
                match timer_arc.lock() {
                    Ok(guard) => *guard,
                    Err(_) => break,
                }
            };
            if !is_running { break; }

            let (repo_path, provider, base_url, model, api_key,
                smart_mode, threshold, push_enabled, push_remote,
                push_branch, commit_prefix, cooldown, last_commit) = {
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
                &commit_prefix, cooldown, last_commit, false, true,
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
pub async fn stop_auto_commit(state: tauri::State<'_, AppState>) -> Result<()> {
    *state.timer_running.lock().map_err(|e| e.to_string())? = false;
    Ok(())
}