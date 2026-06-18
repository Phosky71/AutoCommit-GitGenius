use std::process::Command;

// Quitamos SmartMode (si no se usa) o lo traemos, pero lo mantengo por el run_commit_internal
use crate::config::{
    CommitResult, DiffStatsPublic, LlmProvider, PendingCommit, Result, SmartMode,
};
use crate::llm::call_llm;

// ---------- DIFF STATS (privado al módulo) ----------

#[derive(Clone, Debug)]
pub struct DiffStats {
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub is_significant: bool,
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
    let max_diff = if provider.requires_api_key() {
        16_000
    } else {
        8_000
    };
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

    // BUG FIX #2: El `git status` inicial se ha eliminado. Era propenso a
    // falsos positivos con untracked files vs gitignore.
    // El flujo correcto es: 1) add, 2) diff --cached, 3) si vacío -> abortar.

    // 1. Stage
    if !dry_run {
        let add_status = Command::new("git")
            .args(["add", "."])
            .current_dir(path)
            .status()
            .map_err(|e| format!("Git add error: {}", e))?;

        if !add_status.success() {
            return Err(format!("Git add failed with exit code: {}", add_status));
        }
    }

    // 2. Diff (y chequeo real de vaciado)
    let diff_args = if dry_run {
        vec!["diff"] // Si es dry_run y no hemos hecho add, verificamos working tree
    } else {
        vec!["diff", "--cached"]
    };

    let diff_out = Command::new("git")
        .args(&diff_args)
        .current_dir(path)
        .output()
        .map_err(|e| format!("Git diff error: {}", e))?;

    let diff_content = String::from_utf8_lossy(&diff_out.stdout);

    // BUG FIX #2 (continuación): Si el diff está vacío DESPUÉS del add,
    // significa que realmente no hay cambios para commitear.
    // Esto evita invocar al LLM con un diff vacío o lanzar un commit fallido.
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

    // 4. LLM decision
    let (commit_message, used_llm) = match smart_mode {
        SmartMode::Never => (generate_fallback_message(&stats), false),
        SmartMode::Smart => {
            if stats.is_significant {
                llm_commit_message(provider, base_url, model, api_key, &diff_content, &stats)
                    .await
            } else {
                (generate_fallback_message(&stats), false)
            }
        }
        SmartMode::Always => {
            llm_commit_message(provider, base_url, model, api_key, &diff_content, &stats).await
        }
    };

    // --- FILTRO DE SANITIZACIÓN ---
    let extracted_line = commit_message
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with("```"))
        .unwrap_or(&commit_message);

    let mut clean_message = extracted_line
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .trim()
        .to_string();

    if clean_message.to_lowercase().starts_with("here is") || clean_message.to_lowercase().starts_with("commit message:") {
        if let Some(idx) = clean_message.find(':') {
            clean_message = clean_message[idx + 1..].trim().to_string();
        }
    }

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

    // Lista de ficheros cambiados
    let files_changed_list: Vec<String> = Command::new("git")
        .args(["diff", "--cached", "--name-only"])
        .current_dir(path)
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default();

    // Primeras 60 líneas del diff como preview
    let diff_preview: String = diff_content.lines().take(60).collect::<Vec<_>>().join("\n");

    // Human in the loop → devolver pending antes de commitear
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
    let commit_status = Command::new("git")
        .args(["commit", "-m", &clean_message])
        .current_dir(path)
        .status()
        .map_err(|e| format!("Git commit error: {}", e))?;

    if !commit_status.success() {
        return Err(format!("Git commit failed with exit code: {}", commit_status));
    }

    // 6. Push opcional
    if push_enabled {
        let push_status = Command::new("git")
            .args(["push", push_remote, push_branch])
            .current_dir(path)
            .status()
            .map_err(|e| format!("Git push error: {}", e))?;

        if !push_status.success() {
            // No hacemos Err aquí porque el commit ya se hizo, pero podrías
            // querer añadir un evento de error de push en el futuro.
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