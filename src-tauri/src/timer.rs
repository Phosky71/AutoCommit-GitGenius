use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::time::{interval, Duration};
use tauri::Manager;

use crate::config::{AppState, CommitHistoryEntry, DiffStatsPublic, Result, now_unix, push_history, sanitize_timestamp};
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
        let mut tick = interval(Duration::from_secs(60));
        tick.tick().await;

        loop {
            tick.tick().await;

            let is_running = match timer_arc.lock() {
                Ok(g) => *g,
                Err(_) => break,
            };
            if !is_running { break; }

            let (provider, base_url, model, api_key,
                smart_mode, threshold, hitl, repos, git_token) = { // FIX 1
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
                    c.git_token.clone(),
                )
            };

            let now = now_unix();

            for repo in &repos {
                if !repo.enabled || !repo.timer_enabled { continue; }

                let safe_last_commit = sanitize_timestamp(repo.last_commit_time);
                let elapsed_minutes = (now.saturating_sub(safe_last_commit)) / 60;

                if elapsed_minutes < repo.interval_minutes { continue; }

                let path          = repo.path.clone();
                let trigger_mode  = repo.trigger_mode.clone();

                // --- NUEVO: AI SMART TRIGGER LOGIC ---
                if trigger_mode == "ai" {
                    // Pre-cargamos los cambios para ver el diff real
                    let _ = crate::git::git_command(&path).args(["add", "."]).status();

                    if let Ok(out) = crate::git::git_command(&path).args(["diff", "--cached"]).output() {
                        let diff_content = String::from_utf8_lossy(&out.stdout).to_string();

                        if diff_content.trim().is_empty() {
                            continue; // Nada que commitear
                        }

                        let current_hash = crate::config::calculate_hash(&diff_content);

                        // AHORRO DE TOKENS: Si el diff es idéntico a la última comprobación, saltamos
                        if current_hash == repo.last_checked_diff_hash {
                            continue;
                        }

                        // Actualizamos el hash en el estado global para la próxima vez
                        if let Ok(mut c) = config_arc.lock() {
                            if let Some(r) = c.repos.iter_mut().find(|r| r.id == repo.id) {
                                r.last_checked_diff_hash = current_hash;
                            }
                        }

                        // Preguntamos al LLM si el código está listo
                        let is_ready = crate::llm::ask_llm_if_ready(
                            &provider, &base_url, &model, &api_key, &diff_content
                        ).await.unwrap_or(false);

                        if !is_ready {
                            continue; // La IA decidió que aún no está listo
                        }
                    } else {
                        continue; // Fallo al intentar extraer el diff
                    }
                }
                // --- FIN AI SMART TRIGGER ---

                let push_enabled  = repo.push_enabled;
                let push_remote   = repo.push_remote.clone();
                let push_branch   = repo.push_branch.clone();
                let commit_prefix = repo.commit_prefix.clone();
                let cooldown      = repo.cooldown_minutes;

                match run_commit_internal(
                    &path, &provider, &base_url, &model, &api_key,
                    &smart_mode, threshold, push_enabled,
                    &push_remote, &push_branch,
                    &commit_prefix, cooldown, safe_last_commit, false, hitl,
                    &git_token // FIX 1
                ).await {
                    Ok(r) => {
                        let is_empty = r.message == "No changes to commit"
                            || r.message.starts_with("Cooldown");

                        if !is_empty {
                            if r.pending_approval.is_some() {
                                if let Some(window) = app_handle.get_webview_window("main") {
                                    let _ = window.unminimize();
                                    let _ = window.set_focus();
                                }
                            } else {
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
                                if let Ok(mut c) = config_arc.lock() {
                                    if let Some(re) = c.repos.iter_mut().find(|re| re.path == path) {
                                        re.last_commit_time = now_unix();
                                    }
                                    let _ = push_history(&mut c, entry);
                                }
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