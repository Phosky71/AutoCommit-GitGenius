use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::time::{interval, Duration};
use tauri::Manager;

// Importamos `sanitize_timestamp` que creamos en el archivo anterior
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
                // BUG FIX: Comprobar tanto que el repo exista (enabled)
                // como que su auto-commit esté activo (timer_enabled)
                if !repo.enabled || !repo.timer_enabled { continue; }

                // BUG FIX #3: Sanear el timestamp para evitar bloqueos temporales por desajustes de reloj
                let safe_last_commit = sanitize_timestamp(repo.last_commit_time);

                // Comprobar si ha pasado el intervalo del repo
                let elapsed_minutes = (now.saturating_sub(safe_last_commit)) / 60;
                if elapsed_minutes < repo.interval_minutes { continue; }

                let path          = repo.path.clone();
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
                ).await {
                    Ok(r) => {
                        let is_empty = r.message == "No changes to commit"
                            || r.message.starts_with("Cooldown");

                        if !is_empty {
                            if r.pending_approval.is_some() {
                                // Traer la ventana a primer plano y desminimizar
                                if let Some(window) = app_handle.get_webview_window("main") {
                                    let _ = window.unminimize();
                                    let _ = window.set_focus();
                                }
                            } else {
                                // Flujo normal sin HITL: guardar historial
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

                                    // BUG FIX: Manejar la nueva firma Result<()> de push_history
                                    let _ = push_history(&mut c, entry);
                                }
                            }

                            // SIEMPRE enviamos el evento a la UI.
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