# AutoCommit GitGenius

<div align="center">

<img src="docs/screenshot-placeholder-main.png" alt="AutoCommit GitGenius – main view" width="800">
<!-- Replace with your actual screenshot -->

**Open-source desktop application by [Operia Systems](https://operiasystems.com) for automating Git commits using configurable local or cloud LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri%20v2-24C8D8?logo=tauri)](https://tauri.app)
[![Rust](https://img.shields.io/badge/Backend-Rust-orange?logo=rust)](https://www.rust-lang.org)
[![Latest Release](https://img.shields.io/github/v/release/Phosky71/AutoCommit-GitGenius?label=release)](https://github.com/Phosky71/AutoCommit-GitGenius/releases/latest)
[![Operia Systems](https://img.shields.io/badge/by-Operia%20Systems-6C5DD3)](https://operiasystems.com)

**[⬇ Download for Windows](#download) · [⬇ Download for macOS](#download) · [operiasystems.com](https://operiasystems.com)**

</div>

---

## Table of Contents

- [English](#english)
  - [About Operia Systems](#about-operia-systems)
  - [Overview](#overview)
  - [Download](#download)
  - [Why this project exists](#why-this-project-exists)
  - [Current scope](#current-scope)
  - [Key features](#key-features)
  - [How the commit flow works](#how-the-commit-flow-works)
  - [Supported LLM providers](#supported-llm-providers)
  - [Desktop interface](#desktop-interface)
  - [Architecture](#architecture)
  - [Tech stack](#tech-stack)
  - [Requirements](#requirements)
  - [Installation and development](#installation-and-development)
  - [Build and releases](#build-and-releases)
  - [Configuration and local storage](#configuration-and-local-storage)
  - [Project structure](#project-structure)
  - [Known issues and roadmap](#known-issues-and-roadmap)
  - [Contributing](#contributing)
  - [License](#license)
- [Español](#español)
  - [Sobre Operia Systems](#sobre-operia-systems)
  - [Resumen](#resumen)
  - [Descarga](#descarga)
  - [Por qué existe este proyecto](#por-qué-existe-este-proyecto)
  - [Alcance actual](#alcance-actual)
  - [Funcionalidades principales](#funcionalidades-principales)
  - [Cómo funciona el flujo de commit](#cómo-funciona-el-flujo-de-commit)
  - [Proveedores LLM soportados](#proveedores-llm-soportados)
  - [Interfaz de escritorio](#interfaz-de-escritorio)
  - [Arquitectura](#arquitectura)
  - [Stack tecnológico](#stack-tecnológico)
  - [Requisitos](#requisitos)
  - [Instalación y desarrollo](#instalación-y-desarrollo)
  - [Build y releases](#build-y-releases)
  - [Configuración y almacenamiento local](#configuración-y-almacenamiento-local)
  - [Estructura del proyecto](#estructura-del-proyecto)
  - [Problemas conocidos y hoja de ruta](#problemas-conocidos-y-hoja-de-ruta)
  - [Contribuir](#contribuir)
  - [Licencia](#licencia-1)

---

# English

## About Operia Systems

AutoCommit GitGenius is an open-source project developed and maintained by **[Operia Systems](https://operiasystems.com)**, a software business focused on building practical tools for developers and teams.

This project reflects the Operia Systems approach: lightweight products, honest automation, and real-world developer workflows. It is released under the MIT License so that anyone can use it, improve it, and build on top of it.

> **License:** MIT — Copyright © 2026 Operia Systems. See [LICENSE](LICENSE) for the full text.

---

## Overview

AutoCommit GitGenius is an open-source desktop application built with **Tauri v2**, a **Rust** backend, and a lightweight **HTML/CSS/JavaScript** frontend.

It automates Git commits by watching your repositories for changes, analyzing diffs, generating commit messages with a configurable LLM, and optionally pushing changes to a remote branch — all without leaving your workflow.

The application supports **multiple repositories simultaneously**, each with its own interval, cooldown, prefix, and push settings. It works with local LLM providers such as LM Studio and Ollama, as well as cloud providers such as OpenAI, Gemini, Groq, Anthropic, Mistral, Together, and OpenRouter.

---

## Download

The latest release is **v1.0.7**, available for Windows and macOS:

| Platform | File | Size |
|---|---|---|
| macOS (Apple Silicon) | [AutoCommit_1.0.6_aarch64.dmg](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_1.0.6_aarch64.dmg) | 8.33 MB |
| macOS (app bundle) | [AutoCommit_aarch64.app.tar.gz](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_aarch64.app.tar.gz) | 8.25 MB |
| Windows (installer) | [AutoCommit_1.0.6_x64-setup.exe](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_1.0.6_x64-setup.exe) | 5.09 MB |
| Windows (MSI) | [AutoCommit_1.0.6_x64_en-US.msi](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_1.0.6_x64_en-US.msi) | 7.74 MB |

> **Note:** The release tag is `v1.0.7` but the binary filenames still show `1.0.6` due to a version bump in the tag that was not reflected in `tauri.conf.json` before building. This will be corrected in the next release.

[→ All releases](https://github.com/Phosky71/AutoCommit-GitGenius/releases)

---

## Why this project exists

Many developers work in short iterative cycles and frequently postpone commits because stopping to write a meaningful summary breaks concentration.

AutoCommit GitGenius was created to reduce that friction. Instead of demanding full attention for every commit, it detects what changed, generates a structured message, and asks for confirmation only when the user explicitly wants it.

It is especially useful in local-first setups where developers want Git automation without sending code to external services. With LM Studio or Ollama, the entire process runs entirely offline.

---

## Current scope

AutoCommit GitGenius is focused on **commit automation**. It is not a Git GUI client, a branch management suite, or a merge conflict resolver.

**What it does today:**

- Monitors local Git repositories for changes.
- Analyzes diffs and generates commit messages using heuristics or a configured LLM.
- Runs commit checks on a configurable schedule per repository.
- Optionally requires manual approval before creating the final commit.
- Optionally pushes to a configured remote and branch.
- Stores a local history of generated commits.

**What it does not do:**

- Branch switching or merging.
- Remote repository management.
- Conflict resolution.
- Full Git history visualization.

---

## Key features

### Multi-repository support

Each repository tracked by the application has its own independent settings:

- Commit interval (in minutes).
- Cooldown between consecutive commits (in minutes).
- Custom commit prefix.
- Push enabled or disabled.
- Push remote and branch.
- Enabled or disabled state.

This means you can have one repository auto-committing every 15 minutes and another every two hours, with completely different LLM usage and push rules.

### Configurable LLM providers

The application is provider-agnostic. You can point it at a local server running LM Studio or Ollama with no API key required, or configure it with a cloud provider of your choice. The base URL, model name, and API key are all user-configurable from the Settings panel.

### Three commit generation modes

- `always` — always call the LLM to generate the commit message.
- `smart` — call the LLM only when the diff meets a significance threshold (configurable minimum line count or file count).
- `never` — skip the LLM entirely and use a heuristic commit message based on diff statistics.

### Human-in-the-Loop

When Human-in-the-Loop is enabled, the app stops before creating the final commit and shows a review modal with:

- The generated commit message (editable).
- A diff preview (first 60 lines).
- Diff statistics (files changed, insertions, deletions, estimated tokens).
- A list of changed files.
- An optional Git tag field.
- A push toggle for that specific commit.

Only when the user confirms does the actual `git commit` (and optional `git push`) run.

### Dry run mode

Dry run runs the full analysis and generation pipeline without staging or committing anything. It returns the message that would have been generated along with the diff statistics, so you can test provider settings and message quality before enabling automation.

### Local commit history

Every commit processed by the app is stored locally with:

- Timestamp.
- Repository path.
- Commit message.
- Whether an LLM was used or not.
- Number of files changed.
- Insertions and deletions.
- Estimated token usage.

The history can be filtered by repository and type (AI vs. heuristic), and exported as **CSV** or **JSON**.

### Light and dark theme

The desktop interface supports light and dark themes, with the current selection persisted in configuration.

---

## Screenshots

> Add your screenshots here. Suggested content below.

- **Screenshot 1:** Main repositories panel showing multiple tracked repositories with their status, last commit time, and action buttons.
- **Screenshot 2:** Settings panel showing provider selector, base URL, model name, API key input, Smart Mode selector, threshold slider, and Human-in-the-Loop toggle.
- **Screenshot 3:** Dry run modal showing the generated commit message and diff statistics before any commit is made.
- **Screenshot 4:** Human-in-the-Loop approval modal with editable message, diff preview, changed files list, tag input, and push toggle.
- **Screenshot 5:** Commit history panel with summary stats, repository and type filters, and CSV/JSON export buttons.

---

## How the commit flow works

### Standard automatic flow

1. The background timer ticks every **60 seconds**.
2. For each enabled repository, it checks whether the configured interval has elapsed since the last commit.
3. If the interval has elapsed: run `git status --porcelain` to check for pending changes.
4. If there are changes: run `git add .` to stage everything.
5. Run `git diff --cached` to read the staged diff.
6. Analyze the diff: count insertions, deletions, changed files, and estimate token usage.
7. Based on the configured **Smart Mode**, decide whether to call the LLM.
8. Call the LLM (or generate a heuristic message if the LLM is skipped or fails).
9. Sanitize the LLM response: extract the first non-empty non-markdown line, strip quotes and backticks, remove conversational prefixes.
10. Apply the commit prefix if one is configured.
11. If **Human-in-the-Loop** is enabled: stop, return the pending approval to the frontend, bring the window to the foreground.
12. If HITL is disabled: run `git commit -m "<message>"`.
13. If push is enabled: run `git push <remote> <branch>`.

### Fallback behavior

If the LLM call fails for any reason (connection error, rate limit, invalid response), the app falls back to a heuristic commit message built from diff statistics. This keeps the tool functional even when providers are offline or misconfigured.

### Cooldown

Each repository has a configurable cooldown in minutes. If the time since the last successful commit is less than the cooldown, the commit is skipped for that cycle.

---

## Supported LLM providers

| Provider | Default base URL | Default model | API key required |
|---|---|---|---|
| LM Studio | `http://localhost:1234/v1` | `local-model` | No |
| Ollama | `http://localhost:11434/v1` | `llama3.2` | No |
| LocalAI | `http://localhost:8080/v1` | `gpt-4` | No |
| Custom | `http://localhost:8000/v1` | `local-model` | No |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | Yes |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | Yes |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` | Yes |
| Anthropic | `https://api.anthropic.com/v1` | `claude-3-5-haiku-20241022` | Yes |
| Mistral | `https://api.mistral.ai/v1` | `mistral-small-latest` | Yes |
| Together | `https://api.together.xyz/v1` | `meta-llama/Llama-3-8b-chat-hf` | Yes |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` | Yes |

All cloud providers use an OpenAI-compatible chat completions endpoint except Anthropic, which uses its own `/messages` API with a dedicated request format.

---

## Desktop interface

The UI is currently organized into three main sections accessible from the sidebar:

### Repositories

The primary working view. Shows all tracked repositories with their current state, last commit time, and action buttons for manual commit, dry run, and settings. This is where you add, edit, enable, disable, and remove repositories.

### Settings

Global configuration for the LLM provider (base URL, model, API key, connection test), Smart Mode and significance threshold, default commit prefix, Human-in-the-Loop toggle, and theme selection.

Individual repositories can override the commit prefix set here.

### Commit History

Local commit log with summary statistics (total commits, AI-generated, heuristic, estimated tokens used), filtering by repository and type, and export to CSV or JSON.

---

## Architecture

The project follows a clear desktop application architecture:

┌─────────────────────────────────────────────┐
│ Tauri WebView (UI) │
│ HTML + CSS + JavaScript (main.js) │
└─────────────────────┬───────────────────────┘
│ IPC (Tauri commands)
┌─────────────────────▼───────────────────────┐
│ Rust Backend │
│ commands.rs ─ config.rs ─ git.rs │
│ llm.rs ─ timer.rs ─ main.rs │
└──────────┬──────────────────────┬────────────┘
│ │
┌──────▼──────┐ ┌────────▼────────┐
│ Git (CLI) │ │ LLM Provider │
│ processes │ │ (local/cloud) │
└─────────────┘ └─────────────────┘

- **Frontend** renders through Tauri's embedded WebView with no external browser dependency.
- **Backend** exposes async Rust commands to the frontend via Tauri IPC.
- **Git operations** run as child processes using the system-installed `git` binary.
- **LLM calls** are made over HTTP using `reqwest` with a shared static client.
- **Configuration** is persisted as a local JSON file in the OS config directory.
- **Automation** runs as a background async task with a 60-second tick.

---

## Tech stack

| Layer | Technology |
|---|---|
| Desktop framework | Tauri v2 |
| Backend language | Rust |
| Frontend | HTML, CSS, JavaScript |
| Async runtime | Tokio |
| HTTP client | Reqwest |
| Serialization | Serde + Serde JSON |
| State management | `Arc<Mutex<AppConfig>>` |
| Desktop dialogs | tauri-plugin-dialog |
| CI/CD | GitHub Actions (`tauri-apps/tauri-action`) |
| License | MIT |

---

## Requirements

- **Node.js 20+**
- **Rust stable toolchain** — [install here](https://www.rust-lang.org/tools/install)
- **Git** installed and available in `PATH`
- At least one reachable LLM provider (local or cloud)

---

## Installation and development

```bash
# 1. Clone the repository
git clone https://github.com/Phosky71/AutoCommit-GitGenius.git
cd AutoCommit-GitGenius

# 2. Install frontend dependencies
npm install

# 3. Start the app in development mode
npm run dev
# or
npx tauri dev
```

---

## Build and releases

```bash
npm run build
# or
npx tauri build
```

The installer is generated in `src-tauri/target/release/bundle/`.

The repository includes a GitHub Actions workflow (`.github/workflows/release.yml`) that automatically builds for Windows, macOS, and Ubuntu when a tag matching `v*` is pushed:

```bash
git tag v1.0.8
git push origin v1.0.8
```

The current published release is **v1.0.7**, available for [download here](https://github.com/Phosky71/AutoCommit-GitGenius/releases/latest).

---

## Configuration and local storage

All settings are stored as a single JSON file in the OS user config directory:

| OS | Path |
|---|---|
| Windows | `%APPDATA%\auto-commit-app\config.json` |
| macOS | `~/Library/Application Support/auto-commit-app/config.json` |
| Linux | `~/.config/auto-commit-app/config.json` |

The config file stores global settings, all repository entries, theme preference, and the full commit history. It is created automatically on first run.

> **Important:** API keys are stored in plain text in this local file. Ensure your machine's config directory is appropriately protected.

---

## Project structure

```text
AutoCommit-GitGenius/
├── .github/
│   └── workflows/
│       └── release.yml          # Multi-platform build on v* tags
├── frontend/
│   ├── js/main.js                   # All UI logic, Tauri IPC calls, modals, events
│   ├── index.html               # App shell: sidebar + panels
│   └── style.css                # Design system, dark/light themes, CSS variables
├── src-tauri/
│   ├── src/
│   │   ├── main.rs              # App entry point, Tauri builder, command registration
│   │   ├── commands.rs          # All Tauri commands exposed to the frontend
│   │   ├── config.rs            # AppConfig struct, LlmProvider, SmartMode, persistence
│   │   ├── git.rs               # Diff analysis, fallback messages, commit execution
│   │   ├── llm.rs               # HTTP client for LLM providers, system prompt
│   │   └── timer.rs             # Background automation loop, per-repo scheduling
│   ├── capabilities/            # Tauri v2 permission definitions
│   ├── Cargo.toml               # Rust dependencies
│   └── tauri.conf.json          # App metadata, window config, bundle settings
├── LICENSE                      # MIT License — Copyright © 2026 Operia Systems
├── README.md
├── TODOLIST.MD                  # Active known issues and planned fixes
└── package.json                 # npm scripts: dev, build
```

---

## Known issues and roadmap

This project is actively developed. The following issues are currently known and tracked in [`TODOLIST.MD`](TODOLIST.MD):

| Status | Issue |
|---|---|
| 🔧 In progress | Timer accuracy — current timing behavior is inconsistent and needs correction |
| 🔧 In progress | Token rate limiting — the app does not yet handle provider rate limits gracefully |
| 🔧 In progress | Commit history — some history entries display incorrect or incomplete information |
| 🔧 In progress | API key persistence — the API key is cleared when saving settings from the UI |

These are real issues in the current version. If any of them affect your workflow, contributions are welcome.

> The next immediate priorities are fixing the API key persistence bug and improving timer reliability.

---

## Contributing

Contributions are welcome. The project is open source and maintained by Operia Systems.

1. Fork the repository.
2. Create a descriptive feature or fix branch: `git checkout -b fix/api-key-persistence`.
3. Make focused, well-scoped changes.
4. Write commit messages following [Conventional Commits](https://www.conventionalcommits.org).
5. Open a Pull Request with a clear description of what changed and why.

Good areas to contribute:

- Bug fixes from the TODO list above.
- LLM provider improvements and edge case handling.
- Commit message quality and sanitization.
- Timer reliability and scheduling accuracy.
- UI improvements and usability.
- Documentation and examples.

---

## License

This project is released under the **MIT License**.

> MIT License — Copyright © 2026 Operia Systems
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

See [LICENSE](LICENSE) for the full license text.

---
---

# Español

## Sobre Operia Systems

AutoCommit GitGenius es un proyecto open source desarrollado y mantenido por **[Operia Systems](https://operiasystems.com)**, un negocio de software centrado en construir herramientas prácticas para desarrolladores y equipos.

Este proyecto refleja el enfoque de Operia Systems: productos ligeros, automatización honesta y flujos de trabajo reales para desarrolladores. Se publica bajo la MIT License para que cualquiera pueda usarlo, mejorarlo y construir sobre él.

> **Licencia:** MIT — Copyright © 2026 Operia Systems. Ver [LICENSE](LICENSE) para el texto completo.

---

## Resumen

AutoCommit GitGenius es una aplicación de escritorio **open source** construida con **Tauri v2**, backend en **Rust** y un frontend ligero en **HTML/CSS/JavaScript**.

Automatiza commits de Git vigilando repositorios en busca de cambios, analizando diffs, generando mensajes con un LLM configurable y, opcionalmente, haciendo push a un remote — todo sin interrumpir el flujo de trabajo.

La aplicación soporta **múltiples repositorios simultáneamente**, cada uno con su propio intervalo, cooldown, prefijo y configuración de push. Funciona con proveedores LLM locales como LM Studio y Ollama, y también con proveedores cloud como OpenAI, Gemini, Groq, Anthropic, Mistral, Together y OpenRouter.

---

## Descarga

La última versión es **v1.0.7**, disponible para Windows y macOS:

| Plataforma | Archivo | Tamaño |
|---|---|---|
| macOS (Apple Silicon) | [AutoCommit_1.0.6_aarch64.dmg](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_1.0.6_aarch64.dmg) | 8.33 MB |
| macOS (app bundle) | [AutoCommit_aarch64.app.tar.gz](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_aarch64.app.tar.gz) | 8.25 MB |
| Windows (instalador) | [AutoCommit_1.0.6_x64-setup.exe](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_1.0.6_x64-setup.exe) | 5.09 MB |
| Windows (MSI) | [AutoCommit_1.0.6_x64_en-US.msi](https://github.com/Phosky71/AutoCommit-GitGenius/releases/download/v1.0.7/AutoCommit_1.0.6_x64_en-US.msi) | 7.74 MB |

> **Nota:** El tag del release es `v1.0.7` pero los nombres de los binarios muestran `1.0.6` porque la versión en `tauri.conf.json` no se actualizó antes de compilar. Esto se corregirá en el próximo release.

[→ Todos los releases](https://github.com/Phosky71/AutoCommit-GitGenius/releases)

---

## Por qué existe este proyecto

Muchos desarrolladores trabajan en ciclos cortos e iterativos y suelen retrasar los commits porque parar a escribir un mensaje significativo rompe la concentración.

AutoCommit GitGenius se creó para reducir esa fricción. En lugar de exigir atención completa para cada commit, detecta qué ha cambiado, genera un mensaje estructurado y solo pide confirmación cuando el usuario lo quiere explícitamente.

Es especialmente útil en configuraciones local-first donde se quiere automatización Git sin enviar código a servicios externos. Con LM Studio u Ollama, todo el proceso corre completamente offline.

---

## Alcance actual

AutoCommit GitGenius está centrado en la **automatización de commits**. No es un cliente Git GUI, una suite de gestión de ramas ni una herramienta de resolución de conflictos.

**Lo que hace hoy:**

- Monitorizar repositorios Git locales en busca de cambios.
- Analizar diffs y generar mensajes de commit con heurísticas o un LLM configurado.
- Ejecutar comprobaciones de commit según un horario configurable por repositorio.
- Requerir aprobación manual opcional antes del commit final.
- Hacer push opcional a un remote y branch configurados.
- Guardar un historial local de commits generados.

**Lo que no hace:**

- Cambiar de rama ni hacer merge.
- Gestión de repositorios remotos.
- Resolución de conflictos.
- Visualización completa del historial de Git.

---

## Funcionalidades principales

### Soporte multi-repositorio

Cada repositorio tiene su propia configuración independiente:

- Intervalo de commit (en minutos).
- Cooldown entre commits consecutivos (en minutos).
- Prefijo de commit personalizado.
- Push habilitado o deshabilitado.
- Remote y branch de push.
- Estado habilitado o deshabilitado.

Esto permite, por ejemplo, tener un repositorio haciendo auto-commit cada 15 minutos y otro cada dos horas, con reglas completamente distintas de LLM y push.

### Proveedores LLM configurables

La aplicación es agnóstica al proveedor. Se puede apuntar a un servidor local con LM Studio u Ollama sin necesitar API key, o configurar un proveedor cloud. La base URL, el nombre del modelo y la API key son configurables desde el panel de Settings.

### Tres modos de generación

- `always` — siempre llama al LLM para generar el mensaje.
- `smart` — llama al LLM solo cuando el diff supera un umbral de significatividad (líneas cambiadas o número de archivos).
- `never` — omite el LLM y genera un mensaje heurístico basado en estadísticas del diff.

### Human-in-the-Loop

Cuando está activado, la app se detiene antes del commit final y muestra un modal de revisión con:

- El mensaje generado (editable).
- Un preview del diff (primeras 60 líneas).
- Estadísticas del diff (archivos, inserciones, borrados, tokens estimados).
- Lista de archivos modificados.
- Campo opcional para un tag de Git.
- Toggle de push para ese commit concreto.

Solo cuando el usuario confirma se ejecuta el `git commit` y el `git push` opcional.

### Modo dry run

El dry run ejecuta todo el análisis y la generación sin hacer stage ni commit. Devuelve el mensaje que se habría generado junto con las estadísticas del diff, para poder probar configuraciones y calidad del mensaje antes de activar la automatización.

### Historial local y exportación

Cada commit procesado se guarda localmente con timestamp, ruta del repositorio, mensaje, si se usó LLM o no, número de archivos, inserciones, borrados y uso estimado de tokens.

El historial se puede filtrar por repositorio y tipo, y exportar como **CSV** o **JSON**.

---

## Capturas de pantalla

> Añade tus capturas aquí cuando las tengas. Contenido sugerido:

- **Captura 1:** Panel principal de repositorios con múltiples repos configurados, estado, último commit y botones de acción.
- **Captura 2:** Panel de settings con selector de proveedor, base URL, modelo, API key, Smart Mode y toggle de Human-in-the-Loop.
- **Captura 3:** Modal de dry run mostrando el mensaje generado y estadísticas del diff.
- **Captura 4:** Modal de aprobación con mensaje editable, preview del diff, lista de archivos, campo de tag y toggle de push.
- **Captura 5:** Panel de historial con tarjetas de estadísticas, filtros y botones de exportación CSV/JSON.

---

## Cómo funciona el flujo de commit

### Flujo automático estándar

1. El timer de background hace un tick cada **60 segundos**.
2. Para cada repositorio habilitado, comprueba si ha pasado el intervalo configurado desde el último commit.
3. Si ha pasado: ejecuta `git status --porcelain` para comprobar cambios pendientes.
4. Si hay cambios: ejecuta `git add .` para hacer stage de todo.
5. Ejecuta `git diff --cached` para leer el diff.
6. Analiza el diff: inserciones, borrados, archivos cambiados y estimación de tokens.
7. Según el **Smart Mode** configurado, decide si llamar al LLM.
8. Llama al LLM o genera un mensaje heurístico si el LLM no se usa o falla.
9. Sanea la respuesta: extrae la primera línea válida, elimina markdown, comillas, backticks y prefijos conversacionales.
10. Aplica el prefijo de commit si está configurado.
11. Si **Human-in-the-Loop** está activo: detiene el flujo, devuelve la aprobación pendiente al frontend y trae la ventana al primer plano.
12. Si HITL está desactivado: ejecuta `git commit -m "<mensaje>"`.
13. Si el push está habilitado: ejecuta `git push <remote> <branch>`.

### Fallback

Si la llamada al LLM falla por cualquier motivo (error de conexión, rate limit, respuesta inválida), la app genera un mensaje heurístico a partir de las estadísticas del diff. Esto mantiene la herramienta funcional incluso cuando el proveedor no está disponible o está mal configurado.

### Cooldown

Cada repositorio tiene un cooldown configurable en minutos. Si el tiempo desde el último commit exitoso es menor que el cooldown, ese repositorio se salta en ese ciclo.

---

## Proveedores LLM soportados

| Proveedor | URL base por defecto | Modelo por defecto | API key requerida |
|---|---|---|---|
| LM Studio | `http://localhost:1234/v1` | `local-model` | No |
| Ollama | `http://localhost:11434/v1` | `llama3.2` | No |
| LocalAI | `http://localhost:8080/v1` | `gpt-4` | No |
| Custom | `http://localhost:8000/v1` | `local-model` | No |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | Sí |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | Sí |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` | Sí |
| Anthropic | `https://api.anthropic.com/v1` | `claude-3-5-haiku-20241022` | Sí |
| Mistral | `https://api.mistral.ai/v1` | `mistral-small-latest` | Sí |
| Together | `https://api.together.xyz/v1` | `meta-llama/Llama-3-8b-chat-hf` | Sí |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` | Sí |

Todos los proveedores cloud usan un endpoint compatible con OpenAI para chat completions, excepto Anthropic, que utiliza su propia API `/messages` con un formato de petición dedicado.

---

## Interfaz de escritorio

La UI se organiza en tres secciones principales accesibles desde el sidebar:

### Repositories

Vista principal de trabajo. Muestra todos los repositorios monitorizados con su estado, último commit y botones para commit manual, dry run y edición. Aquí se añaden, editan, habilitan, deshabilitan y eliminan repositorios.

### Settings

Configuración global del proveedor LLM (base URL, modelo, API key, test de conexión), Smart Mode y umbral de significatividad, prefijo de commit por defecto, toggle de Human-in-the-Loop y selección de tema.

Los repositorios individuales pueden sobreescribir el prefijo configurado aquí.

### Commit History

Historial local con estadísticas resumidas (commits totales, generados por IA, heurísticos, tokens estimados), filtros por repositorio y tipo, y exportación a CSV o JSON.

---

## Arquitectura

┌─────────────────────────────────────────────┐
│ Tauri WebView (UI) │
│ HTML + CSS + JavaScript (main.js) │
└─────────────────────┬───────────────────────┘
│ IPC (comandos Tauri)
┌─────────────────────▼───────────────────────┐
│ Backend Rust │
│ commands.rs ─ config.rs ─ git.rs │
│ llm.rs ─ timer.rs ─ main.rs │
└──────────┬──────────────────────┬────────────┘
│ │
┌──────▼──────┐ ┌────────▼────────┐
│ Git (CLI) │ │ Proveedor LLM │
│ procesos │ │ (local/cloud) │
└─────────────┘ └─────────────────┘

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework de escritorio | Tauri v2 |
| Lenguaje backend | Rust |
| Frontend | HTML, CSS y JavaScript |
| Runtime async | Tokio |
| Cliente HTTP | Reqwest |
| Serialización | Serde + Serde JSON |
| Gestión de estado | `Arc<Mutex<AppConfig>>` |
| Diálogos de escritorio | tauri-plugin-dialog |
| CI/CD | GitHub Actions (`tauri-apps/tauri-action`) |
| Licencia | MIT |

---

## Requisitos

- **Node.js 20+**
- **Rust stable** — [instalar aquí](https://www.rust-lang.org/tools/install)
- **Git** instalado y disponible en `PATH`
- Al menos un proveedor LLM accesible (local o cloud)

---

## Instalación y desarrollo

```bash
# 1. Clonar el repositorio
git clone https://github.com/Phosky71/AutoCommit-GitGenius.git
cd AutoCommit-GitGenius

# 2. Instalar dependencias del frontend
npm install

# 3. Ejecutar en modo desarrollo
npm run dev
# o
npx tauri dev
```

---

## Build y releases

```bash
npm run build
# o
npx tauri build
```

El instalador se genera en `src-tauri/target/release/bundle/`.

El repositorio incluye un workflow de GitHub Actions (`.github/workflows/release.yml`) que compila automáticamente para Windows, macOS y Ubuntu cuando se sube un tag con formato `v*`:

```bash
git tag v1.0.8
git push origin v1.0.8
```

El release publicado actualmente es **v1.0.7**, disponible para [descarga aquí](https://github.com/Phosky71/AutoCommit-GitGenius/releases/latest).

---

## Configuración y almacenamiento local

Toda la configuración se guarda en un único archivo JSON en el directorio de configuración del usuario:

| SO | Ruta |
|---|---|
| Windows | `%APPDATA%\auto-commit-app\config.json` |
| macOS | `~/Library/Application Support/auto-commit-app/config.json` |
| Linux | `~/.config/auto-commit-app/config.json` |

El archivo almacena ajustes globales, todos los repositorios configurados, preferencia de tema e historial completo de commits. Se crea automáticamente en el primer uso.

> **Importante:** Las API keys se almacenan en texto plano en este archivo local. Asegúrate de que el directorio de configuración de tu equipo está apropiadamente protegido.

---

## Estructura del proyecto

```text
AutoCommit-GitGenius/
├── .github/
│   └── workflows/
│       └── release.yml          # Build multiplataforma en tags v*
├── frontend/
│   ├── js/main.js                   # Lógica completa de UI, llamadas IPC, modales, eventos
│   ├── index.html               # Shell de la app: sidebar + paneles
│   └── style.css                # Sistema de diseño, temas dark/light, variables CSS
├── src-tauri/
│   ├── src/
│   │   ├── main.rs              # Punto de entrada, builder de Tauri, registro de comandos
│   │   ├── commands.rs          # Todos los comandos Tauri expuestos al frontend
│   │   ├── config.rs            # AppConfig, LlmProvider, SmartMode, persistencia
│   │   ├── git.rs               # Análisis de diff, mensajes heurísticos, ejecución de commits
│   │   ├── llm.rs               # Cliente HTTP para proveedores LLM, system prompt
│   │   └── timer.rs             # Bucle de automatización en background, scheduling por repo
│   ├── capabilities/            # Definiciones de permisos de Tauri v2
│   ├── Cargo.toml               # Dependencias Rust
│   └── tauri.conf.json          # Metadatos de la app, ventana, configuración de bundle
├── LICENSE                      # MIT License — Copyright © 2026 Operia Systems
├── README.md
├── TODOLIST.MD                  # Problemas conocidos activos y fixes planificados
└── package.json                 # Scripts npm: dev, build
```

---

## Problemas conocidos y hoja de ruta

Este proyecto está en desarrollo activo. Los siguientes problemas son conocidos y están registrados en [`TODOLIST.MD`](TODOLIST.MD):

| Estado | Problema |
|---|---|
| 🔧 En progreso | Precisión del timer — el comportamiento actual es inconsistente y necesita corrección |
| 🔧 En progreso | Rate limiting — la app no maneja aún de forma robusta los límites de cuota de los proveedores |
| 🔧 En progreso | Historial de commits — algunas entradas muestran información incorrecta o incompleta |
| 🔧 En progreso | Persistencia de la API key — la clave se borra al guardar cambios en Settings |

Estos son problemas reales en la versión actual. Si alguno afecta a tu flujo de trabajo, las contribuciones son bienvenidas.

> Las próximas prioridades inmediatas son corregir el bug de la API key y mejorar la fiabilidad del timer.

---

## Contribuir

Las contribuciones son bienvenidas. El proyecto es open source y está mantenido por Operia Systems.

1. Haz fork del repositorio.
2. Crea una rama descriptiva: `git checkout -b fix/api-key-persistence`.
3. Haz cambios bien acotados y enfocados.
4. Escribe mensajes de commit siguiendo [Conventional Commits](https://www.conventionalcommits.org).
5. Abre una Pull Request con una descripción clara de qué cambió y por qué.

Áreas especialmente útiles para contribuir:

- Bugs del TODO list.
- Mejoras y casos edge en proveedores LLM.
- Calidad y sanitización de mensajes de commit.
- Fiabilidad y precisión del timer.
- Mejoras de UI y usabilidad.
- Documentación y ejemplos.

---

## Licencia

Este proyecto se distribuye bajo la **MIT License**.

> MIT License — Copyright © 2026 Operia Systems
>
> Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y los archivos de documentación asociados (el "Software"), para utilizar el Software sin restricción, incluyendo sin limitación los derechos de usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del Software, y permitir a las personas a quienes se les proporcione el Software hacer lo mismo, sujeto a las siguientes condiciones:
>
> El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o partes sustanciales del Software.
>
> EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO.

Ver [LICENSE](LICENSE) para el texto completo.
