# CommitCraft 🚀

<div align="center">

![CommitCraft Banner](https://img.shields.io/badge/CommitCraft-AI%20Git%20Automation-blue?style=for-the-badge)
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![Tauri](https://img.shields.io/badge/tauri-%2324C8DB.svg?style=for-the-badge&logo=tauri&logoColor=%23FFFFFF)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**AI-Powered Git Commit Automation with Google Gemini**

CommitCraft is a cross-platform desktop application that revolutionizes your Git workflow by automatically generating professional, conventional commit messages using Google's Gemini 2.0 Flash AI model.

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Configuration](#-configuration) •
[Development](#-development) •
[License](#-license)

</div>

---

## ✨ Features

### 🤖 **AI-Powered Commit Messages**
- Leverages Google Gemini 2.0 Flash for instant, intelligent commit message generation
- Follows Conventional Commits specification automatically
- Analyzes git diffs to understand code changes contextually
- RAG (Retrieval-Augmented Generation) implementation for consistent, professional messaging

### ⏰ **Automated Commit Scheduling**
- Configure custom intervals (minutes) for automatic commits
- Background timer system with real-time status updates
- Smart detection: only commits when changes are detected
- Auto-start option on application launch

### 🔐 **Secure Configuration**
- Encrypted API key storage in system config directory
- Persistent settings across sessions
- Test API key functionality before saving
- Safe credential management

### 🎨 **Modern User Interface**
- Clean, gradient-based design
- Responsive layout with real-time feedback
- Directory browser for easy repository selection
- Visual status indicators for commit operations

### 🚀 **Cross-Platform Support**
- Windows, macOS, and Linux compatible
- Native performance with Tauri framework
- Low resource consumption
- System tray integration ready

---

## 📋 Prerequisites

Before installing CommitCraft, ensure you have:

1. **Git** installed and configured
   ```bash
   git --version
   ```

2. **Google Gemini API Key**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key (free tier available)
   - Keep it handy for configuration

3. **System Requirements**
   - Windows 10/11, macOS 10.15+, or Linux
   - 100MB free disk space
   - Active internet connection for AI features

---

## 🚀 Installation

### Option 1: Download Pre-built Binary (Recommended)

1. Go to [Releases](https://github.com/yourusername/commitcraft/releases)
2. Download the appropriate version for your OS:
   - **Windows**: `CommitCraft_x.x.x_x64.msi` or `.exe`
   - **macOS**: `CommitCraft_x.x.x_x64.dmg`
   - **Linux**: `CommitCraft_x.x.x_amd64.deb` or `.AppImage`
3. Install and launch the application

### Option 2: Build from Source

#### Prerequisites for Building
- [Rust](https://rustup.rs/) (latest stable version)
- [Node.js](https://nodejs.org/) 16+ (optional, for frontend dev)
- Platform-specific dependencies:
  - **Linux**: `libwebkit2gtk-4.0-dev`, `build-essential`, `curl`, `wget`, `libssl-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`
  - **macOS**: Xcode Command Line Tools
  - **Windows**: Microsoft C++ Build Tools

#### Build Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/commitcraft.git
cd commitcraft

# Install Rust dependencies
cargo build --release

# Run the application
cargo tauri dev

# Build for production
cargo tauri build
```

The compiled binary will be in `src-tauri/target/release/`

---

## 🎯 Usage

### First-Time Setup

1. **Launch CommitCraft**
   - Open the application from your applications menu or run the executable

2. **Configure Gemini API Key**
   ```
   🔑 Gemini API Configuration section:
   - Paste your API key in the password field
   - Click "Test" to verify connectivity
   - Status message confirms successful validation
   ```

3. **Select Repository**
   ```
   📁 Repository section:
   - Click "Browse" to select your Git repository folder
   - Or manually enter the path
   - Configuration auto-saves
   ```

4. **Configure Auto-Commit (Optional)**
   ```
   ⚙️ Auto-Commit Settings:
   - Set interval in minutes (default: 30)
   - Check "Start automatically on app launch" if desired
   ```

### Manual Commit

1. Make changes to your repository files
2. Click **"Manual Commit"** button
3. CommitCraft will:
   - Stage all changes (`git add .`)
   - Analyze the diff
   - Generate an AI commit message
   - Commit with the generated message
   - Push to remote repository

**Example Output:**
```
✅ Commit successful: feat(auth): implement JWT token validation
```

### Automatic Commits

1. Configure your repository and API key
2. Set desired interval (e.g., 30 minutes)
3. Click **"Start Auto-Commit"**
4. CommitCraft runs in the background:
   - Checks for changes every X minutes
   - Auto-commits and pushes if changes detected
   - Displays notifications for each operation
5. Click **"Stop"** to halt automatic commits

---

## ⚙️ Configuration

### Configuration File Location

CommitCraft stores settings in:
- **Windows**: `C:\Users\{username}\AppData\Roaming\auto-commit-app\config.json`
- **macOS**: `/Users/{username}/Library/Application Support/auto-commit-app/config.json`
- **Linux**: `/home/{username}/.config/auto-commit-app/config.json`

### Configuration Schema

```json
{
  "repo_path": "/path/to/your/repository",
  "auto_commit_enabled": false,
  "interval_minutes": 30,
  "auto_start": false,
  "gemini_api_key": "AIza..."
}
```

### Manual Configuration

You can manually edit the `config.json` file, but it's recommended to use the UI for automatic validation.

---

## 🤖 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  CommitCraft UI                     │
│              (HTML/CSS/JavaScript)                  │
└────────────────────┬────────────────────────────────┘
                     │ Tauri IPC
┌────────────────────▼────────────────────────────────┐
│               Rust Backend                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  State Management (Arc<Mutex<Config>>)       │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  Git Operations (git2 crate)                 │  │
│  │  - Repository scanning                       │  │
│  │  - Staging changes                           │  │
│  │  - Diff extraction                           │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  AI Integration (Gemini API)                 │  │
│  │  - System context (RAG)                      │  │
│  │  - Prompt engineering                        │  │
│  │  - Response parsing                          │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  Scheduler (Tokio async runtime)             │  │
│  │  - Interval-based execution                  │  │
│  │  - Event emission                            │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Commit Generation Process

1. **Change Detection**
   ```rust
   git2::Repository::open(path)
   repo.statuses() // Check for modifications
   ```

2. **Staging**
   ```bash
   git add .
   ```

3. **Diff Analysis**
   ```bash
   git diff --cached --stat
   git diff --cached
   ```

4. **AI Processing**
   ```
   System Context (RAG) + Git Diff → Gemini 2.0 Flash
   ↓
   Conventional Commit Message
   ```

5. **Commit & Push**
   ```bash
   git commit -m "feat(component): description"
   git push
   ```

### RAG System Context

CommitCraft uses a comprehensive system instruction for Gemini:

```
- Expert in Conventional Commits specification
- Analyzes code changes contextually
- Follows format: <type>(<scope>): <subject>
- Types: feat, fix, docs, style, refactor, test, chore, perf
- Maximum 50 characters
- Imperative mood
- Specific and actionable
```

**Example transformations:**

| Git Diff | Generated Commit |
|----------|------------------|
| `+function validateToken()` | `feat(auth): add JWT token validation` |
| `- const API_URL = 'old'` | `refactor(config): update API endpoint URL` |
| `+ README.md installation steps` | `docs(readme): add installation instructions` |

---

## 🛠️ Development

### Project Structure

```
commitcraft/
├── src-tauri/
│   ├── src/
│   │   └── main.rs          # Rust backend logic
│   ├── Cargo.toml           # Rust dependencies
│   ├── tauri.conf.json      # Tauri configuration
│   └── icons/               # Application icons
├── frontend/
│   └── index.html           # UI interface
├── LICENSE                  # MIT License
└── README.md               # This file
```

### Key Dependencies

**Rust:**
- `tauri`: Desktop framework
- `git2`: Git operations
- `reqwest`: HTTP client for Gemini API
- `tokio`: Async runtime
- `serde`: Serialization
- `dirs`: System directories

**Frontend:**
- Pure HTML/CSS/JavaScript (no framework required)
- Tauri IPC for backend communication

### Running in Development Mode

```bash
# Terminal 1: Watch Rust backend
cd src-tauri
cargo watch -x 'tauri dev'

# Auto-reloads on file changes
```

### Building for Production

```bash
# All platforms
cargo tauri build

# Specific platform
cargo tauri build --target x86_64-pc-windows-msvc    # Windows
cargo tauri build --target x86_64-apple-darwin       # macOS Intel
cargo tauri build --target aarch64-apple-darwin      # macOS Apple Silicon
cargo tauri build --target x86_64-unknown-linux-gnu  # Linux
```

### Testing

```bash
# Run Rust tests
cargo test

# Test API integration
cargo run --bin test_gemini
```

---

## 🔒 Security

### API Key Protection
- Stored in system config directory (not in repository)
- Never logged or transmitted except to Gemini API
- Masked in UI (password field)
- Recommend using environment variables for CI/CD

### Best Practices
- Never commit `config.json` to version control
- Rotate API keys periodically
- Use read-only Git credentials when possible
- Review generated commits before auto-push (optional setting)

---

## 🐛 Troubleshooting

### Common Issues

**"Gemini API Key not configured"**
- Solution: Add your API key in the 🔑 Gemini API Configuration section
- Verify key at [Google AI Studio](https://aistudio.google.com/app/apikey)

**"No folder was selected"**
- Solution: Ensure you click "Browse" and select a valid Git repository
- Repository must have `.git` folder

**"Failed to open repository"**
- Solution: Path must point to a Git repository root
- Run `git init` if not initialized

**"Timer is already running"**
- Solution: Stop the current auto-commit before starting a new one
- Restart the application if stuck

**Build errors on Linux**
- Solution: Install required dependencies:
  ```bash
  sudo apt-get install libwebkit2gtk-4.0-dev \
    build-essential curl wget libssl-dev \
    libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev
  ```

---

## 🗺️ Roadmap

- [ ] Multi-repository management
- [ ] Commit history viewer
- [ ] Custom commit templates
- [ ] Branch selection
- [ ] Rollback functionality
- [ ] System tray minimization
- [ ] Dark/Light theme toggle
- [ ] Offline mode with local AI models
- [ ] GitHub/GitLab integration
- [ ] Commit analytics dashboard

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows Rust style guidelines (`cargo fmt`)
- All tests pass (`cargo test`)
- Commit messages follow Conventional Commits

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

```
Copyright (c) 2025 Antonio Juan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
[...]
```

---

## 👤 Author

**Antonio Juan**
- Location: Cartagena, Región de Murcia, España
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- [Tauri](https://tauri.app/) - For the amazing desktop framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) - For the powerful AI model
- [git2-rs](https://github.com/rust-lang/git2-rs) - For Git integration
- [Conventional Commits](https://www.conventionalcommits.org/) - For commit standards

---

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/commitcraft?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/commitcraft?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/commitcraft)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/commitcraft)

---

<div align="center">

**Made with ❤️ and 🦀 Rust**

If CommitCraft helps your workflow, consider giving it a ⭐!

</div>
