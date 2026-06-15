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
    {
        let mut running = state.timer_running.lock().map_err(|e| e.to_string())?;
        if *running { return Err("Timer is already running".into()); }
        *running = true;
    }

    let timer_arc  = Arc::clone(&state.timer_running);
    let config_arc = Arc::clone(&state.config);

    tauri::async_runtime::spawn(async move {
        let mut tick = interval(Duration::from_secs(60)); // tick cada minuto
        tick.tick().await; // skip tick inmediato

        loop {
            tick.tick().await;

            let is_running = match timer_arc.lock() {
                Ok(g) => *g,
                Err(_) => break,
            };
            if !is_running { break; }

            // Leer configuración global + lista de repos
            let (provider, base_url, model, api_key,
                smart_mode, threshold, hitl, repos) = {
                let Ok(c) = config_arc.lock() else { continue; };
                (
                    c.provider.clone(),
                    c.llm_base_url.clone(),
                    c.llm_model_name.clone(),
                    c.llm_api_key.clone(),
                    c.smart_mode.clone(),
                    c.smart_threshold_lines,
                    c.human_in_the_loop,
                    c.repos.clone(),
                )
            };

            let now = now_unix();

            for repo in &repos {
                // Solo repos activos
                if !repo.enabled { continue; }

                // Comprobar si ha pasado el intervalo del repo
                let elapsed_minutes = (now.saturating_sub(repo.last_commit_time)) / 60;
                if elapsed_minutes < repo.interval_minutes as u64 { continue; }

                let path          = repo.path.clone();
                let push_enabled  = repo.push_enabled;
                let push_remote   = repo.push_remote.clone();
                let push_branch   = repo.push_branch.clone();
                let commit_prefix = repo.commit_prefix.clone();
                let cooldown      = repo.cooldown_minutes;
                let last_commit   = repo.last_commit_time;

                match run_commit_internal(
                    &path, &provider, &base_url, &model, &api_key,
                    &smart_mode, threshold, push_enabled,
                    &push_remote, &push_branch,
                    &commit_prefix, cooldown, last_commit, false, hitl,
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
                                repo_path: path.clone(),
                                message: r.message.clone(),
                                used_llm: r.used_llm,
                                files_changed: stats.files_changed,
                                insertions: stats.insertions,
                                deletions: stats.deletions,
                                estimated_tokens: stats.estimated_tokens,
                            };
                            // Actualizar last_commit_time del repo
                            if let Ok(mut c) = config_arc.lock() {
                                if let Some(re) = c.repos.iter_mut().find(|re| re.path == path) {
                                    re.last_commit_time = now_unix();
                                }
                                push_history(&mut c, entry);
                            }
                            let _ = app_handle.emit("commit-status", &r);
                        }
                    }
                    Err(e) => { let _ = app_handle.emit("commit-error", &e); }
                }
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