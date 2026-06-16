# AutoCommit GitGenius

<div align="center">

**Aplicación de escritorio desarrollada por [Operia Systems](https://operiasystems.com) que automatiza los commits de Git usando IA.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri%20v2-24C8D8?logo=tauri)](https://tauri.app)
[![Rust](https://img.shields.io/badge/Backend-Rust-orange?logo=rust)](https://www.rust-lang.org)

</div>

---

## ¿Qué es AutoCommit GitGenius?

AutoCommit GitGenius es una aplicación de escritorio open-source construida con **Tauri v2** (backend en Rust) y un frontend en **Vanilla JS/HTML/CSS**. Su propósito es automatizar los commits de Git analizando el `git diff` y generando un mensaje de commit siguiendo el estándar **Conventional Commits** mediante un LLM configurable.

El LLM por defecto es **LM Studio** (local), pero soporta multitud de proveedores cloud y locales.

---

## Funcionalidades reales

### Gestión de repositorios
- Añade múltiples repositorios Git, cada uno con su propia configuración
- Cada repo tiene: intervalo de commit automático (minutos), cooldown, prefijo de commit, remote/branch de push, y si el push está habilitado
- Los repos pueden habilitarse o deshabilitarse individualmente desde la UI

### Generación de mensajes de commit
El flujo real en `git.rs` es:
1. `git status --porcelain` — comprueba si hay cambios
2. `git add .` — stagea todos los cambios
3. `git diff --cached` — obtiene el diff
4. Analiza el diff: cuenta inserciones, borrados, ficheros cambiados y estima tokens
5. Según el **Smart Mode** configurado, decide si llamar al LLM:
   - `Always` — siempre llama al LLM
   - `Smart` — llama al LLM solo si el diff supera el umbral de líneas (por defecto: 10) o tiene ≥3 ficheros
   - `Never` — genera un mensaje heurístico sin LLM
6. Si el LLM falla, usa un mensaje de fallback heurístico basado en las stats del diff
7. Sanitiza la respuesta del LLM (elimina backticks, comillas, texto conversacional)
8. Aplica el prefijo de commit si está configurado
9. Hace `git commit -m "<mensaje>"`
10. Si push está habilitado: `git push <remote> <branch>`

### Proveedores LLM soportados

El proveedor se configura en la UI. Los disponibles (definidos en `config.rs`) son:

| Proveedor | URL por defecto | Modelo por defecto |
|-----------|----------------|--------------------|
| LM Studio | `http://localhost:1234/v1` | `local-model` |
| Ollama | `http://localhost:11434/v1` | `llama3.2` |
| LocalAI | `http://localhost:8080/v1` | `gpt-4` |
| Custom | `http://localhost:8000/v1` | `local-model` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| Anthropic | `https://api.anthropic.com/v1` | `claude-3-5-haiku-20241022` |
| Mistral | `https://api.mistral.ai/v1` | `mistral-small-latest` |
| Together | `https://api.together.xyz/v1` | `meta-llama/Llama-3-8b-chat-hf` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |

Los proveedores locales (LM Studio, Ollama, LocalAI, Custom) **no requieren API key**. El resto sí.

### Timer automático (`timer.rs`)
- El timer corre en background con un tick cada **60 segundos**
- Por cada tick revisa todos los repos habilitados
- Comprueba si ha pasado el `interval_minutes` configurado para ese repo
- Si hay cambios, ejecuta el flujo de commit completo
- Emite eventos `commit-status` y `commit-error` al frontend vía Tauri
- Si **Human-in-the-Loop** está activo y hay un commit pendiente, trae la ventana a primer plano

### Human-in-the-Loop (HITL)
Cuando está activo, en lugar de hacer el commit directamente:
- Muestra al usuario el mensaje generado, el diff preview (primeras 60 líneas), las stats y la lista de ficheros
- El usuario puede aprobar, editar el mensaje, añadir un Git tag y decidir si hacer push
- El commit solo se realiza cuando el usuario confirma con `confirm_commit`

### Dry Run
- Ejecuta el análisis del diff y la generación del mensaje **sin hacer ningún commit**
- Devuelve el mensaje con prefijo `[DRY RUN]` junto con las stats del diff

### Historial de commits
- Registra cada commit: timestamp, ruta del repo, mensaje, si usó LLM, ficheros cambiados, inserciones, borrados y tokens estimados
- Filtrable por repositorio y tipo (AI vs heurístico)
- Exportable como **CSV**
- Persistido en el archivo de configuración local

### Configuración persistente
La configuración se guarda en JSON en el directorio de configuración del sistema operativo:
- **Windows**: `%APPDATA%\auto-commit-app\config.json`
- **macOS**: `~/Library/Application Support/auto-commit-app/config.json`
- **Linux**: `~/.config/auto-commit-app/config.json`

---

## Interfaz de usuario

La UI tiene tres secciones principales (sidebar):

- **Repositories** — lista de repos configurados, botón de commit manual, dry run, preview del diff y gestión de repos
- **Settings** — configuración del proveedor LLM (URL, modelo, API key, test de conexión), Smart Mode, umbrales, comportamiento de commit, tema
- **Commit History** — historial de commits con filtros y exportación

Soporta tema **dark/light** con persistencia en la configuración.

---

## Stack tecnológico

| Capa | Tecnología |
|------|------------|
| Framework de escritorio | Tauri v2 |
| Backend | Rust (tokio, reqwest, serde, once_cell) |
| Frontend | Vanilla JavaScript, HTML, CSS |
| Fuente de tipografía | Satoshi (Google Fonts) |
| CI/CD | GitHub Actions (`tauri-apps/tauri-action`) |

---

## Prerrequisitos

- [Node.js](https://nodejs.org) v20+
- [Rust](https://www.rust-lang.org/tools/install) (toolchain stable)
- [Git](https://git-scm.com) instalado y accesible desde el PATH
- Un proveedor LLM accesible (local o cloud)

---

## Instalación y desarrollo

### 1. Clonar el repositorio

```bash
git clone https://github.com/Phosky71/AutoCommit-GitGenius.git
cd AutoCommit-GitGenius
```

### 2. Instalar dependencias del frontend

```bash
npm install
```

### 3. Ejecutar en modo desarrollo

```bash
npm run dev
# o
npx tauri dev
```

### 4. Compilar para producción

```bash
npm run build
# o
npx tauri build
```

El instalador se genera en `src-tauri/target/release/bundle/`.

---

## Estructura del proyecto

```
AutoCommit-GitGenius/
├── frontend/
│   ├── app.js          # Lógica completa de la UI (navegación, modales, llamadas Tauri)
│   ├── index.html      # Shell de la app (sidebar + paneles: repos, settings, history)
│   └── style.css       # Estilos (tema dark/light, variables CSS)
├── src-tauri/
│   ├── src/
│   │   ├── main.rs         # Entry point, registro de comandos Tauri
│   │   ├── commands.rs     # Comandos expuestos al frontend vía IPC
│   │   ├── config.rs       # Structs de configuración, AppState, LlmProvider enum
│   │   ├── git.rs          # Lógica de git: diff, commit, push, fallback
│   │   ├── llm.rs          # Cliente HTTP para LLM (OpenAI-compatible + Anthropic)
│   │   └── timer.rs        # Timer background para commits automáticos por repo
│   ├── capabilities/   # Permisos de Tauri v2
│   ├── Cargo.toml      # Dependencias Rust
│   └── tauri.conf.json # Configuración de la app (id: com.operiasystems.auto-commit)
├── .github/
│   └── workflows/
│       └── release.yml     # Build multiplataforma en push de tag v*
├── LICENSE             # MIT — Copyright (c) 2026 OPERIA SYSTEMS
└── package.json        # Scripts: tauri, dev, build
```

---

## Releases

El workflow `release.yml` se ejecuta automáticamente al subir un tag con formato `v*` (ej: `v1.0.7`). Compila la aplicación en tres plataformas:

- **Windows** (`windows-latest`)
- **macOS** (`macos-latest`)
- **Linux/Ubuntu** (`ubuntu-22.04`)

Los binarios se publican como draft en GitHub Releases para revisión antes de publicarlos.

---

## Contribuir

1. Haz fork del repositorio
2. Crea una rama: `git checkout -b feat/tu-feature`
3. Haz tus cambios y commitea siguiendo [Conventional Commits](https://www.conventionalcommits.org)
4. Abre un Pull Request

---

## Licencia

MIT License — Copyright (c) 2026 OPERIA SYSTEMS. Ver [LICENSE](LICENSE).

---

## Sobre Operia Systems

[Operia Systems](https://operiasystems.com) es una empresa tecnológica que desarrolla herramientas prácticas para desarrolladores y negocios. AutoCommit GitGenius es uno de sus proyectos open-source.

- Web: [operiasystems.com](https://operiasystems.com)

---

<div align="center">
  <sub>Desarrollado por <a href="https://operiasystems.com">Operia Systems</a></sub>
</div>
