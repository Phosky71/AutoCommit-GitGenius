# AutoCommit GitGenius

<div align="center">

**Open-source desktop app by [Operia Systems](https://operiasystems.com) for automating Git commits with configurable local or cloud LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri%20v2-24C8D8?logo=tauri)](https://tauri.app)
[![Rust](https://img.shields.io/badge/Backend-Rust-orange?logo=rust)](https://www.rust-lang.org)
[![Release](https://img.shields.io/github/v/release/Phosky71/AutoCommit-GitGenius)](https://github.com/Phosky71/AutoCommit-GitGenius/releases)

</div>

---

## Table of contents

- [English](#english)
  - [Overview](#overview)
  - [Why this project exists](#why-this-project-exists)
  - [Current scope](#current-scope)
  - [Key features](#key-features)
  - [How the commit flow works](#how-the-commit-flow-works)
  - [Supported LLM providers](#supported-llm-providers)
  - [Desktop interface](#desktop-interface)
  - [Architecture](#architecture)
  - [Tech stack](#tech-stack)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Development](#development)
  - [Build and releases](#build-and-releases)
  - [Configuration and local storage](#configuration-and-local-storage)
  - [Project structure](#project-structure)
  - [Roadmap and known issues](#roadmap-and-known-issues)
  - [Contributing](#contributing)
  - [License](#license)
  - [About Operia Systems](#about-operia-systems)
- [Español](#español)
  - [Resumen](#resumen)
  - [Por qué existe este proyecto](#por-qué-existe-este-proyecto)
  - [Alcance actual](#alcance-actual)
  - [Funcionalidades principales](#funcionalidades-principales)
  - [Cómo funciona el flujo de commit](#cómo-funciona-el-flujo-de-commit)
  - [Proveedores LLM soportados](#proveedores-llm-soportados)
  - [Interfaz de escritorio](#interfaz-de-escritorio)
  - [Arquitectura](#arquitectura)
  - [Stack tecnológico](#stack-tecnológico)
  - [Requisitos](#requisitos)
  - [Instalación](#instalación)
  - [Desarrollo](#desarrollo)
  - [Build y releases](#build-y-releases)
  - [Configuración y almacenamiento local](#configuración-y-almacenamiento-local)
  - [Estructura del proyecto](#estructura-del-proyecto)
  - [Hoja de ruta y problemas conocidos](#hoja-de-ruta-y-problemas-conocidos)
  - [Contribuir](#contribuir)
  - [Licencia](#licencia-1)
  - [Sobre Operia Systems](#sobre-operia-systems)

---

# English

## Overview

AutoCommit GitGenius is an open-source desktop application built with **Tauri v2**, a **Rust** backend, and a lightweight **HTML/CSS/JavaScript** frontend.

Its purpose is to reduce the friction of repetitive Git commits by detecting repository changes, generating commit messages, and optionally pushing updates to a configured remote branch.

The project comes from **Operia Systems** and is intended as a practical developer productivity tool rather than a full Git replacement.

---

## Why this project exists

Many developers work in short iterative cycles and often postpone commits because stopping to summarize changes feels like overhead.

AutoCommit GitGenius aims to make that workflow smoother by helping with commit generation while still giving the user control over when and how commits are created.

It is especially useful in local-first setups where developers want to combine Git automation with local LLMs such as LM Studio or Ollama, while still keeping cloud provider support available when needed.

---

## Current scope

AutoCommit GitGenius is focused on **commit automation**, not on replacing Git clients or full repository hosting workflows.

Today, the project is designed around:

- Monitoring local Git repositories.
- Detecting changed files and analyzing diffs.
- Generating commit messages using either heuristics or a configurable LLM.
- Optionally requiring manual approval before creating the final commit.
- Optionally pushing commits to a configured remote and branch.
- Storing local commit history for later review and export.

What it is **not**:

- A full Git GUI client.
- A merge conflict resolution tool.
- A repository hosting platform.
- A complete branch orchestration system.

---

## Key features

### Multi-repository management

The application supports multiple Git repositories, each with its own local settings.

Each repository can store:

- Commit interval in minutes.
- Cooldown time between commits.
- Commit prefix.
- Push enabled or disabled.
- Push remote.
- Push branch.
- Enabled state.

This makes it possible to use different automation rules for different projects.

### Configurable LLM integration

The app does not depend on a single AI provider.

Instead, it supports both local and cloud-based LLM providers, allowing users to choose between privacy-first local setups and hosted APIs depending on their workflow.

### Smart commit generation modes

AutoCommit GitGenius includes three commit generation modes:

- `always` — always call the LLM.
- `smart` — call the LLM only when the diff is considered significant.
- `never` — skip the LLM and generate a heuristic commit message.

This gives the user a practical balance between speed, cost, determinism, and output quality.

### Human-in-the-Loop review

For users who want automation without losing control, the project includes an approval flow before the final commit is created.

When enabled, the app can show:

- The generated commit message.
- A short diff preview.
- Basic diff statistics.
- The list of changed files.
- An optional tag field.
- A push toggle for the approval step.

### Dry run mode

Dry run allows the user to preview the generated message and the analyzed diff stats without creating a real commit.

This is useful for testing provider settings, checking message quality, or validating automation behavior before enabling it in a real workflow.

### Local history and export

The application stores a local history of generated commits including repository path, timestamp, message, whether an LLM was used, file count, insertions, deletions, and estimated tokens.

This history can be filtered in the UI and exported as CSV.

---

## Screenshot placeholders

Add your own screenshots in this section later.

- **Screenshot placeholder:** Main dashboard with multiple repositories configured and visible in the repositories panel.
- **Screenshot placeholder:** Settings screen showing provider configuration, Smart Mode selection, threshold settings, and Human-in-the-Loop toggle.
- **Screenshot placeholder:** Dry run modal displaying a generated commit message and diff statistics.
- **Screenshot placeholder:** Approval modal showing editable commit message, tag field, push toggle, and diff preview.
- **Screenshot placeholder:** Commit history view with statistics cards, filters, and export buttons.

---

## How the commit flow works

The current core workflow is intentionally simple and practical.

### Standard flow

1. Check whether the repository has pending changes.
2. Stage changes with `git add .`.
3. Read the diff.
4. Analyze insertions, deletions, changed files, and estimated token usage.
5. Decide whether to call the LLM based on the configured Smart Mode.
6. If the LLM is used, sanitize the response to remove markdown, quotes, or conversational filler.
7. Apply a commit prefix if one is configured.
8. Create the Git commit.
9. Optionally push to the configured remote and branch.

### Fallback behavior

If the LLM fails, the app falls back to a heuristic commit message based on the diff statistics.

That behavior is important because it keeps the tool usable even when a provider is offline, unavailable, rate-limited, or incorrectly configured.

### Approval flow

When Human-in-the-Loop is enabled, the app stops before creating the final commit and returns a pending approval object instead.

The user can then review the proposed message, edit it, add a tag, decide whether to push, and confirm the operation manually.

---

## Supported LLM providers

The configuration model currently defines support for the following providers:

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

This provider flexibility is one of the most important characteristics of the project and should be reflected in its public presentation.

---

## Desktop interface

The desktop UI is currently organized into three main sections:

### Repositories

This section is focused on adding repositories, editing repository-specific settings, enabling or disabling tracked repositories, running manual commits, and previewing diffs.

### Settings

This section centralizes LLM provider settings, base URL, model name, API key, Smart Mode, threshold configuration, commit behavior, and theme selection.

### Commit History

This section shows the local commit log with summary stats, filters, and export actions.

The interface also supports dark and light themes.

---

## Architecture

AutoCommit GitGenius follows a straightforward desktop application architecture:

- **Frontend:** HTML, CSS, and JavaScript UI rendered through Tauri.
- **Backend:** Rust commands exposed to the frontend through Tauri IPC.
- **Git execution layer:** Native Git commands executed through system processes.
- **LLM integration layer:** HTTP-based provider calls for OpenAI-compatible and Anthropic-style APIs.
- **Persistence layer:** Local JSON configuration stored in the user config directory.
- **Automation layer:** Background timer that checks enabled repositories on a fixed interval.

This architecture keeps the project lightweight, portable, and easy to understand for contributors.

---

## Tech stack

| Layer | Technology |
|---|---|
| Desktop framework | Tauri v2 |
| Backend | Rust |
| Frontend | HTML, CSS, JavaScript |
| Async runtime | Tokio |
| HTTP client | Reqwest |
| Serialization | Serde / Serde JSON |
| Desktop dialogs | tauri-plugin-dialog |
| CI/CD | GitHub Actions |
| License | MIT |

---

## Requirements

Before running the project locally, make sure you have:

- **Node.js 20+**
- **Rust stable**
- **Git** installed and available in `PATH`
- A reachable LLM provider, either local or cloud-based

Depending on your chosen provider, you may also need an API key.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Phosky71/AutoCommit-GitGenius.git
cd AutoCommit-GitGenius
```

### Install dependencies

```bash
npm install
```

### Start in development mode

```bash
npm run dev
```

You can also run:

```bash
npx tauri dev
```

---

## Development

The project is intended to be approachable for developers familiar with frontend JavaScript and Rust.

Typical development areas include:

- Improving commit generation heuristics.
- Expanding provider compatibility.
- Refining UI and desktop UX.
- Hardening repository validation and error handling.
- Improving configuration persistence.
- Enhancing release automation.

If you are contributing, it is a good idea to keep changes scoped and focused, especially when touching both frontend and backend logic.

---

## Build and releases

To build a production version locally:

```bash
npm run build
```

You can also use:

```bash
npx tauri build
```

The repository includes a GitHub Actions release workflow that runs on tags matching the `v*` pattern.

Current public release version:

- **v1.0.7**

The workflow is configured to build for:

- Windows
- macOS
- Ubuntu Linux

Release artifacts are published as GitHub releases.

---

## Configuration and local storage

The application persists configuration to a local JSON file in the system config directory.

Typical paths are:

- **Windows:** `%APPDATA%\auto-commit-app\config.json`
- **macOS:** `~/Library/Application Support/auto-commit-app/config.json`
- **Linux:** `~/.config/auto-commit-app/config.json`

The stored configuration includes global settings, repository entries, theme, and commit history.

Because the project stores provider configuration locally, users should still review how they manage secrets on their own machines.

---

## Project structure

```text
AutoCommit-GitGenius/
├── .github/
│   └── workflows/
│       └── release.yml
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── src-tauri/
│   ├── src/
│   │   ├── commands.rs
│   │   ├── config.rs
│   │   ├── git.rs
│   │   ├── llm.rs
│   │   ├── main.rs
│   │   └── timer.rs
│   ├── Cargo.toml
│   └── tauri.conf.json
├── LICENSE
├── README.md
├── TODOLIST.MD
└── package.json
```

### File roles

- `frontend/index.html` — main application shell.
- `frontend/app.js` — client-side application logic and UI behavior.
- `frontend/style.css` — visual design and theme system.
- `src-tauri/src/main.rs` — Tauri application entry point.
- `src-tauri/src/commands.rs` — backend commands exposed to the frontend.
- `src-tauri/src/config.rs` — configuration model and persistence helpers.
- `src-tauri/src/git.rs` — diff analysis and commit execution logic.
- `src-tauri/src/llm.rs` — provider communication layer.
- `src-tauri/src/timer.rs` — background automation loop.

---

## Roadmap and known issues

This project is usable, but it is still evolving.

The repository currently includes the following TODO items:

- Fix timer accuracy and current timing issues.
- Handle token request limits and rate limiting more safely.
- Fix incorrect or incomplete history information.
- Fix API key saving so it is not cleared when saving changes.

Keeping these items visible in the README is valuable because it sets realistic expectations for users and contributors.

---

## Contributing

Contributions are welcome.

A practical contribution workflow is:

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Use clear commit messages, ideally following Conventional Commits.
5. Open a Pull Request with a concise explanation of what changed and why.

Useful contribution areas include:

- Bug fixes
- UX improvements
- LLM provider enhancements
- Commit history improvements
- Timer reliability
- Documentation and examples

---

## License

This project is released under the **MIT License**.

See [LICENSE](LICENSE) for the full text.

---

## About Operia Systems

[Operia Systems](https://operiasystems.com) is the business behind this project.

AutoCommit GitGenius is one of its open-source developer tools and reflects a practical approach to software: lightweight products, useful automation, and real-world developer workflows.

---

# Español

## Resumen

AutoCommit GitGenius es una aplicación de escritorio **open source** construida con **Tauri v2**, backend en **Rust** y un frontend ligero en **HTML/CSS/JavaScript**.

Su objetivo es reducir la fricción de los commits repetitivos detectando cambios en repositorios, generando mensajes de commit y, si se desea, haciendo push a un remote y una branch configurados.

El proyecto proviene de **Operia Systems** y está planteado como una herramienta práctica de productividad para desarrolladores, no como un sustituto completo de Git.

---

## Por qué existe este proyecto

Muchos desarrolladores trabajan en ciclos cortos e iterativos y suelen retrasar los commits porque parar a resumir cambios rompe el ritmo.

AutoCommit GitGenius busca hacer ese flujo más cómodo ayudando con la generación del commit, pero manteniendo el control del usuario sobre cuándo y cómo se realiza.

Resulta especialmente útil en configuraciones local-first donde se quiere combinar automatización Git con LLMs locales como LM Studio u Ollama, sin perder compatibilidad con proveedores cloud.

---

## Alcance actual

AutoCommit GitGenius está centrado en la **automatización de commits**, no en sustituir clientes Git completos ni flujos enteros de hosting de repositorios.

Actualmente está diseñado para:

- Monitorizar repositorios Git locales.
- Detectar archivos modificados y analizar diffs.
- Generar mensajes de commit mediante heurísticas o un LLM configurable.
- Requerir aprobación manual opcional antes del commit final.
- Hacer push opcional a un remote y una branch configurados.
- Guardar historial local para revisión y exportación.

Lo que **no** es:

- Un cliente Git GUI completo.
- Una herramienta de resolución de conflictos.
- Una plataforma de hosting de repositorios.
- Un sistema completo de orquestación de ramas.

---

## Funcionalidades principales

### Gestión multi-repositorio

La aplicación soporta múltiples repositorios Git, cada uno con su propia configuración local.

Cada repo puede almacenar:

- Intervalo de commit en minutos.
- Cooldown entre commits.
- Prefijo de commit.
- Push activado o desactivado.
- Remote de push.
- Branch de push.
- Estado habilitado.

Esto permite usar reglas distintas según el proyecto.

### Integración configurable con LLMs

La aplicación no depende de un único proveedor de IA.

Soporta proveedores locales y cloud, permitiendo elegir entre privacidad en entorno local o APIs alojadas según el flujo de trabajo de cada usuario.

### Modos de generación inteligentes

AutoCommit GitGenius incluye tres modos:

- `always` — siempre llama al LLM.
- `smart` — solo llama al LLM cuando el diff se considera significativo.
- `never` — no usa LLM y genera un mensaje heurístico.

Esto da un equilibrio práctico entre velocidad, coste, determinismo y calidad del resultado.

### Human-in-the-Loop

Para quienes quieren automatización sin perder control, el proyecto incluye un flujo de aprobación antes de crear el commit final.

Cuando está activo, la app puede mostrar:

- El mensaje generado.
- Un diff preview corto.
- Estadísticas básicas del diff.
- La lista de archivos modificados.
- Un campo opcional para tag.
- Un selector para decidir si hacer push.

### Modo dry run

El dry run permite previsualizar el mensaje generado y las estadísticas del diff sin crear un commit real.

Es útil para probar la configuración del proveedor, revisar la calidad del mensaje o validar el comportamiento de la automatización antes de activarla en un flujo real.

### Historial local y exportación

La aplicación guarda un historial local de commits generados con ruta del repositorio, timestamp, mensaje, si se usó LLM, número de archivos, inserciones, borrados y tokens estimados.

Ese historial se puede filtrar desde la interfaz y exportar como CSV.

---

## Marcadores para capturas

Añade aquí tus capturas reales más adelante.

- **Marcador de captura:** Pantalla principal con varios repositorios configurados en el panel de repositories.
- **Marcador de captura:** Pantalla de settings mostrando proveedor, Smart Mode, umbral y Human-in-the-Loop.
- **Marcador de captura:** Modal de dry run con mensaje generado y estadísticas del diff.
- **Marcador de captura:** Modal de aprobación con mensaje editable, campo de tag, push toggle y diff preview.
- **Marcador de captura:** Vista de commit history con tarjetas de estadísticas, filtros y botones de exportación.

---

## Cómo funciona el flujo de commit

El flujo principal actual está planteado para ser sencillo y práctico.

### Flujo estándar

1. Comprueba si el repositorio tiene cambios pendientes.
2. Hace stage con `git add .`.
3. Lee el diff.
4. Analiza inserciones, borrados, archivos cambiados y estimación de tokens.
5. Decide si llamar al LLM según el Smart Mode configurado.
6. Si usa el LLM, sanea la respuesta para eliminar markdown, comillas o texto conversacional.
7. Aplica un prefijo si está configurado.
8. Crea el commit.
9. Opcionalmente hace push al remote y branch configurados.

### Comportamiento de fallback

Si el LLM falla, la aplicación usa un mensaje heurístico basado en las estadísticas del diff.

Ese comportamiento es importante porque mantiene la herramienta utilizable incluso cuando el proveedor está caído, no disponible, limitado por cuota o mal configurado.

### Flujo con aprobación

Cuando Human-in-the-Loop está activado, la aplicación se detiene antes del commit final y devuelve una aprobación pendiente.

Después, el usuario puede revisar el mensaje propuesto, editarlo, añadir un tag, decidir si quiere push y confirmar manualmente la operación.

---

## Proveedores LLM soportados

El modelo de configuración actual define soporte para estos proveedores:

| Proveedor | URL base por defecto | Modelo por defecto | Requiere API key |
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

Esta flexibilidad con proveedores es una de las características más importantes del proyecto y conviene reflejarla bien en su presentación pública.

---

## Interfaz de escritorio

La interfaz se organiza actualmente en tres secciones principales.

### Repositories

Esta sección se centra en añadir repositorios, editar ajustes por repo, habilitar o deshabilitar repos monitorizados, lanzar commits manuales y previsualizar diffs.

### Settings

Aquí se centralizan la configuración del proveedor LLM, la base URL, el modelo, la API key, el Smart Mode, el umbral, el comportamiento del commit y el tema visual.

### Commit History

Aquí se muestra el historial local con estadísticas resumidas, filtros y acciones de exportación.

La interfaz también soporta tema oscuro y claro.

---

## Arquitectura

AutoCommit GitGenius sigue una arquitectura de aplicación de escritorio clara y bastante directa:

- **Frontend:** interfaz en HTML, CSS y JavaScript renderizada con Tauri.
- **Backend:** comandos en Rust expuestos al frontend por IPC de Tauri.
- **Capa Git:** ejecución de comandos Git nativos mediante procesos del sistema.
- **Capa LLM:** llamadas HTTP a proveedores compatibles con OpenAI y Anthropic.
- **Persistencia:** configuración JSON local en el directorio de configuración del usuario.
- **Automatización:** timer en background que revisa repos habilitados en intervalos fijos.

Esta arquitectura mantiene el proyecto ligero, portable y fácil de entender para contribuidores.

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework de escritorio | Tauri v2 |
| Backend | Rust |
| Frontend | HTML, CSS y JavaScript |
| Runtime async | Tokio |
| Cliente HTTP | Reqwest |
| Serialización | Serde / Serde JSON |
| Diálogos de escritorio | tauri-plugin-dialog |
| CI/CD | GitHub Actions |
| Licencia | MIT |

---

## Requisitos

Antes de ejecutar el proyecto en local, asegúrate de tener:

- **Node.js 20+**
- **Rust stable**
- **Git** instalado y disponible en `PATH`
- Un proveedor LLM accesible, ya sea local o cloud

Según el proveedor elegido, también puede hacer falta una API key.

---

## Instalación

### Clonar el repositorio

```bash
git clone https://github.com/Phosky71/AutoCommit-GitGenius.git
cd AutoCommit-GitGenius
```

### Instalar dependencias

```bash
npm install
```

### Ejecutar en desarrollo

```bash
npm run dev
```

También puedes usar:

```bash
npx tauri dev
```

---

## Desarrollo

El proyecto está pensado para ser accesible para desarrolladores que se mueven bien entre JavaScript frontend y Rust.

Las áreas de trabajo más habituales incluyen:

- Mejorar heurísticas de generación de commits.
- Ampliar compatibilidad con proveedores.
- Refinar la UI y la UX de escritorio.
- Endurecer la validación de repositorios y el manejo de errores.
- Mejorar la persistencia de configuración.
- Mejorar la automatización de releases.

Si contribuyes, es buena idea mantener los cambios bien acotados, sobre todo cuando tocan frontend y backend a la vez.

---

## Build y releases

Para generar una versión de producción en local:

```bash
npm run build
```

También puedes usar:

```bash
npx tauri build
```

El repositorio incluye un workflow de GitHub Actions que se ejecuta con tags que sigan el patrón `v*`.

Versión pública actual:

- **v1.0.7**

El workflow está preparado para compilar en:

- Windows
- macOS
- Ubuntu Linux

Los artefactos se publican en GitHub Releases.

---

## Configuración y almacenamiento local

La aplicación guarda la configuración en un archivo JSON local dentro del directorio de configuración del sistema.

Rutas típicas:

- **Windows:** `%APPDATA%\auto-commit-app\config.json`
- **macOS:** `~/Library/Application Support/auto-commit-app/config.json`
- **Linux:** `~/.config/auto-commit-app/config.json`

La configuración almacenada incluye ajustes globales, repositorios, tema e historial local de commits.

Como la app guarda configuración del proveedor en local, conviene que cada usuario revise cómo maneja sus secretos en su propio equipo.

---

## Estructura del proyecto

```text
AutoCommit-GitGenius/
├── .github/
│   └── workflows/
│       └── release.yml
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── src-tauri/
│   ├── src/
│   │   ├── commands.rs
│   │   ├── config.rs
│   │   ├── git.rs
│   │   ├── llm.rs
│   │   ├── main.rs
│   │   └── timer.rs
│   ├── Cargo.toml
│   └── tauri.conf.json
├── LICENSE
├── README.md
├── TODOLIST.MD
└── package.json
```

### Rol de archivos

- `frontend/index.html` — shell principal de la aplicación.
- `frontend/app.js` — lógica cliente y comportamiento de la UI.
- `frontend/style.css` — diseño visual y sistema de temas.
- `src-tauri/src/main.rs` — punto de entrada de Tauri.
- `src-tauri/src/commands.rs` — comandos backend expuestos al frontend.
- `src-tauri/src/config.rs` — modelo de configuración y persistencia.
- `src-tauri/src/git.rs` — análisis de diff y ejecución de commits.
- `src-tauri/src/llm.rs` — capa de comunicación con proveedores.
- `src-tauri/src/timer.rs` — bucle de automatización en background.

---

## Hoja de ruta y problemas conocidos

El proyecto es utilizable, pero sigue evolucionando.

El repositorio incluye actualmente estos TODOs:

- Corregir la precisión y el comportamiento actual de los timers.
- Manejar mejor los límites de peticiones y rate limiting.
- Corregir información incorrecta o incompleta en el historial.
- Arreglar el guardado de la API key para que no se borre al guardar cambios.

Mantener estos puntos visibles en el README es útil porque fija expectativas realistas para usuarios y contribuidores.

---

## Contribuir

Las contribuciones son bienvenidas.

Un flujo práctico para contribuir sería:

1. Hacer fork del repositorio.
2. Crear una rama de trabajo.
3. Hacer un cambio bien acotado.
4. Usar mensajes de commit claros, idealmente con Conventional Commits.
5. Abrir una Pull Request con una explicación breve de qué cambió y por qué.

Áreas especialmente útiles para contribuir:

- Corrección de bugs
- Mejoras de UX
- Mejoras de proveedores LLM
- Mejoras en historial
- Fiabilidad del timer
- Documentación y ejemplos

---

## Licencia

Este proyecto se distribuye bajo la **MIT License**.

Consulta [LICENSE](LICENSE) para el texto completo.

---

## Sobre Operia Systems

[Operia Systems](https://operiasystems.com) es el negocio detrás de este proyecto.

AutoCommit GitGenius es una de sus herramientas open source para desarrolladores y refleja un enfoque práctico del software: productos ligeros, automatización útil y flujos reales de trabajo para developers.
