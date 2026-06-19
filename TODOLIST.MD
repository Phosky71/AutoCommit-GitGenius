# TODO List



- [ ] **Migrar comandos a Tokio (Async):** Reemplazar `std::process::Command` por `tokio::process::Command` en las llamadas a Git. Esto hará que la ejecución sea 100% asíncrona y evitará que la interfaz (UI) se congele durante operaciones largas como `git push`.

- [ ] **Actualizar README.MD**