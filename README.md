# AutoCommit GitGenius

<div align="center">

**A desktop application by [Operia Systems](https://operiasystems.com) that automates Git commit messages using AI.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri%20v2-24C8D8?logo=tauri)](https://tauri.app)
[![Rust](https://img.shields.io/badge/Backend-Rust-orange?logo=rust)](https://www.rust-lang.org)
[![Powered by Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.0%20Flash-4285F4?logo=google)](https://deepmind.google/technologies/gemini/)

</div>

---

## What is AutoCommit GitGenius?

AutoCommit GitGenius is an open-source desktop application developed by **Operia Systems** that integrates AI into your daily Git workflow. Instead of writing commit messages manually, the app analyzes your staged changes and automatically generates clear, professional commit messages following conventional commit standards — powered by **Google Gemini 2.0 Flash**.

The application is built with **Tauri v2** (Rust backend) and a vanilla JavaScript/HTML/CSS frontend, making it lightweight, fast, and cross-platform.

---

## Features

- **AI-Generated Commit Messages** — Analyzes your `git diff` and generates semantic commit messages via Google Gemini 2.0 Flash
- **Scheduled Automatic Commits** — Set up a timer to commit automatically at defined intervals
- **Multi-Repository Management** — Add and manage multiple Git repositories from a single interface
- **Branch Management** — View and switch branches directly from the app
- **Diff Preview** — Inspect staged changes before committing
- **Smart Mode** — AI decides the commit type and scope automatically
- **Dark / Light Theme** — Toggle between themes with persistent preferences
- **Human-in-the-Loop** — Optionally review and approve each AI-generated message before committing

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Desktop Framework | [Tauri v2](https://tauri.app) |
| Backend | Rust |
| Frontend | Vanilla JavaScript, HTML, CSS |
| AI Model | Google Gemini 2.0 Flash |
| HTTP Client | `reqwest` (Rust) |
| Async Runtime | `tokio` |

---

## Prerequisites

Before building or running the app, make sure you have the following installed:

- [Node.js](https://nodejs.org) (v18+)
- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- [Tauri CLI](https://tauri.app/start/): `npm install -g @tauri-apps/cli`
- A valid **Google Gemini API key** (get one at [Google AI Studio](https://aistudio.google.com))

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Phosky71/AutoCommit-GitGenius.git
cd AutoCommit-GitGenius
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Run in development mode

```bash
npm run tauri dev
```

### 4. Build for production

```bash
npm run tauri build
```

The compiled installer will be located in `src-tauri/target/release/bundle/`.

---

## Configuration

On first launch, the app will prompt you to configure:

- **Gemini API Key** — Required for AI-generated commit messages
- **Default repository path** — The Git repository to work with by default

Configuration is stored locally on your machine.

---

## Project Structure

```
AutoCommit-GitGenius/
├── frontend/           # Vanilla JS/HTML/CSS frontend
│   ├── app.js          # Main application logic
│   ├── index.html      # App shell
│   └── style.css       # Styles
├── src-tauri/          # Rust backend (Tauri)
│   ├── src/            # Rust source files
│   ├── capabilities/   # Tauri permission definitions
│   ├── Cargo.toml      # Rust dependencies
│   └── tauri.conf.json # Tauri configuration
├── .github/workflows/  # CI/CD pipelines
├── LICENSE             # MIT License
└── package.json        # Node.js manifest
```

---

## Contributing

Contributions are welcome! If you find a bug or want to suggest a new feature:

1. Fork the repository
2. Create a new branch: `git checkout -b feat/your-feature`
3. Make your changes and commit them
4. Open a Pull Request

Please follow the [Conventional Commits](https://www.conventionalcommits.org) specification for commit messages.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## About Operia Systems

[Operia Systems](https://operiasystems.com) is a technology company building practical tools for developers and businesses. Our products focus on seamless integration into existing workflows — from IDE security plugins to developer automation utilities.

- Website: [operiasystems.com](https://operiasystems.com)
- Product: [Operia Security](https://operiasystems.com) — Real-time vulnerability detection for JetBrains IDEs

---

<div align="center">
  <sub>Built with care by <a href="https://operiasystems.com">Operia Systems</a></sub>
</div>
