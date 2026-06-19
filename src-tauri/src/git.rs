use std::process::Command;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use crate::config::{
    CommitResult, DiffStatsPublic, LlmProvider, PendingCommit, Result, SmartMode,
};
use crate::llm::call_llm;

// ---------- HELPER PARA OCULTAR VENTANA DE CONSOLA EN WINDOWS ----------
pub fn git_command(path: &str) -> Command {
    let mut cmd = Command::new("git");
    cmd.current_dir(path);

    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    cmd
}

// ---------- DIFF STATS (privado al módulo) ----------

#[derive(Clone, Debug)]
pub struct DiffStats {
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub is_significant: bool,
    pub is_too_large: bool, // FIX: Nuevo cortocircuito de seguridad
    pub estimated_tokens: usize,
}

pub fn analyze_diff(diff: &str, threshold_lines: u64) -> DiffStats {
    let mut insertions = 0usize;
    let mut deletions = 0usize;
    let mut files_changed = 0usize;
    for line in diff.lines() {
        if line.starts_with('+') && !line.starts_with("+++") {
            insertions += 1;
        } else if line.starts_with('-') && !line.starts_with("---") {
            deletions += 1;
        } else if line.starts_with("diff --git") {
            files_changed += 1;
        }
    }
    let total = (insertions + deletions) as u64;
    let prompt_overhead_tokens = 95;

    DiffStats {
        files_changed,
        insertions,
        deletions,
        is_significant: total >= threshold_lines || files_changed >= 3,
        // FIX: Cortocircuito. Si hay más de 40 archivos o 3000 líneas, evitamos el LLM.
        is_too_large: files_changed > 40 || total > 3000,
        estimated_tokens: (diff.len() / 4) + prompt_overhead_tokens,
    }
}

pub fn generate_fallback_message(stats: &DiffStats) -> String {
    if stats.files_changed == 0 {
        return "chore: minor update".to_string();
    }
    if stats.deletions > stats.insertions * 2 {
        return format!(
            "refactor: remove code across {} file(s)",
            stats.files_changed
        );
    }
    if stats.insertions > 0 && stats.deletions == 0 {
        return format!("feat: add new code in {} file(s)", stats.files_changed);
    }
    format!(
        "chore: update {} file(s) (+{} -{} lines)",
        stats.files_changed, stats.insertions, stats.deletions
    )
}

// ---------- LLM MESSAGE HELPER ----------

pub async fn llm_commit_message(
    provider: &LlmProvider,
    base_url: &str,
    model: &str,
    api_key: &str,
    diff_content: &str,
    stats: &DiffStats,
) -> (String, bool) {
    let max_diff = if provider.requires_api_key() { 16_000 } else { 8_000 };
    let diff_text = if diff_content.len() > max_diff {
        format!("(Diff truncated)...\n{}", &diff_content[..max_diff])
    } else {
        diff_content.to_string()
    };
    let prompt = format!(
        "Generate a commit message for these changes:\n\n{}",
        diff_text
    );
    match call_llm(provider, base_url, model, api_key, &prompt).await {
        Ok(msg) => (msg, true),
        Err(_) => (generate_fallback_message(stats), false),
    }
}

// ---------- CORE COMMIT LOGIC ----------

#[allow(clippy::too_many_arguments)]
pub async fn run_commit_internal(
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
    git_token: &str,
) -> Result<CommitResult> {
    use crate::config::now_unix;

    // Cooldown check
    if !dry_run && cooldown_minutes > 0 {
        let elapsed = now_unix().saturating_sub(last_commit_time);
        if elapsed < cooldown_minutes * 60 {
            return Ok(CommitResult {
                message: format!(
                    "Cooldown active: {}s remaining",
                    cooldown_minutes * 60 - elapsed
                ),
                used_llm: false,
                diff_stats: None,
                pending_approval: None,
            });
        }
    }

    // 1. Stage (solo si no es dry run)
    if !dry_run {
        let add_status = git_command(path).args(["add", "."]).status()
            .map_err(|e| format!("Git add error: {}", e))?;

        if !add_status.success() {
            return Err(format!("Git add failed with exit code: {}", add_status));
        }
    }

    // 2. Diff (FIX: Manejo del Dry Run para detectar cambios staged + unstaged)
    let diff_args = if dry_run {
        // Comprobamos si HEAD existe (si no es el primer commit del repo)
        let has_head = git_command(path).args(["rev-parse", "--verify", "HEAD"])
            .output().map(|o| o.status.success()).unwrap_or(false);

        if has_head { vec!["diff", "HEAD"] } else { vec!["diff"] }
    } else {
        vec!["diff", "--cached"]
    };

    let diff_out = git_command(path).args(&diff_args).output()
        .map_err(|e| format!("Git diff error: {}", e))?;

    let diff_content = String::from_utf8_lossy(&diff_out.stdout);

    if diff_content.trim().is_empty() {
        return Ok(CommitResult {
            message: "No changes to commit".to_string(),
            used_llm: false,
            diff_stats: None,
            pending_approval: None,
        });
    }

    // 3. Analyze
    let stats = analyze_diff(&diff_content, smart_threshold_lines);
    let stats_public = DiffStatsPublic {
        files_changed: stats.files_changed,
        insertions: stats.insertions,
        deletions: stats.deletions,
        estimated_tokens: stats.estimated_tokens,
    };

    // 4. LLM decision (FIX: Cortocircuito si es demasiado masivo)
    let (commit_message, used_llm) = if stats.is_too_large {
        (generate_fallback_message(&stats), false)
    } else {
        match smart_mode {
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
        }
    };

    // --- FILTRO DE SANITIZACIÓN ---
    let extracted_line = commit_message.lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with("```"))
        .unwrap_or(&commit_message);

    let mut clean_message = extracted_line.trim_matches(|c| c == '"' || c == '\'' || c == '`').trim().to_string();

    if clean_message.to_lowercase().starts_with("here is") || clean_message.to_lowercase().starts_with("commit message:") {
        if let Some(idx) = clean_message.find(':') {
            clean_message = clean_message[idx + 1..].trim().to_string();
        }
    }

    if !commit_prefix.is_empty() {
        clean_message = format!("{} {}", commit_prefix.trim(), clean_message);
    }

    if dry_run {
        return Ok(CommitResult {
            message: format!("[DRY RUN] {}", clean_message),
            used_llm,
            diff_stats: Some(stats_public),
            pending_approval: None,
        });
    }

    let files_changed_list: Vec<String> = git_command(path)
        .args(["diff", "--cached", "--name-only"]).output()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
        .unwrap_or_default();

    let diff_preview: String = diff_content.lines().take(60).collect::<Vec<_>>().join("\n");

    if human_in_the_loop {
        return Ok(CommitResult {
            message: clean_message.clone(),
            used_llm,
            diff_stats: Some(stats_public.clone()),
            pending_approval: Some(PendingCommit {
                message: clean_message.clone(),
                used_llm,
                diff_stats: stats_public,
                diff_preview,
                files_changed_list,
            }),
        });
    }

    // 5. Commit
    let commit_status = git_command(path).args(["commit", "-m", &clean_message]).status()
        .map_err(|e| format!("Git commit error: {}", e))?;

    if !commit_status.success() {
        return Err(format!("Git commit failed with exit code: {}", commit_status));
    }

    // 6. Push opcional
    if push_enabled {
        let mut target_url = push_remote.to_string();

        if let Ok(out) = git_command(path).args(["remote", "get-url", push_remote]).output() {
            let url = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !url.is_empty() { target_url = url; }
        }

        if !git_token.is_empty() && target_url.starts_with("https://") {
            target_url = target_url.replacen("https://", &format!("https://{}@", git_token), 1);
        }

        let push_status = git_command(path)
            .args(["-c", "credential.helper=", "push", &target_url, push_branch]).status()
            .map_err(|e| format!("Git push error: {}", e))?;

        if !push_status.success() {
            println!("Warning: Push failed, but commit was successful");
        }
    }

    Ok(CommitResult {
        message: clean_message,
        used_llm,
        diff_stats: Some(stats_public),
        pending_approval: None,
    })
}