#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod llm;
mod git;
mod commands;
mod timer;

use config::AppState;
use commands::*;
use timer::{start_auto_commit, stop_auto_commit};

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
            open_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}