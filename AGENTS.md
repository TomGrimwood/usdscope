# usdscope

A general library: a USD stage drawn fast for editing (`read`: plain arrays from pxr;
`view`: a viewport over Newton's OpenGL viewer). `README.md` has the modules.

## Rules

- **General, not any one user's.** Nothing here names an application or a use; an app's
  tools (what a click means, its panels) are its own, given to `Viewport` as callbacks.
- **Plain arrays between the two.** `read` imports neither Newton nor a window; `view` draws
  what `read` returns and arrays the app gives it.
- **Fast on big scenes.** A mesh is read and uploaded once and drawn at every copy; what moves
  is placed per frame, only when its time changed; nothing per frame in Python per prim.
- **Newton's viewer as it comes.** Use its public calls; where an app's input must come first
  (mouse, keys), wrap its handlers on the instance, never patch its files.
- **Errors are said** in the terminal (and a log file if given), never only in a panel.
- Comments and docstrings say plainly what the code does, for a reader with only the file.
- New capability is a function or method with keyword arguments.
- Work on a branch; merge to `main` with `--no-ff` when `make` passes. Commit only when asked.

## Commands

```bash
uv sync
make        # lint, format, types, the tests (the viewport's need a GPU; skipped without)
make fmt
uv run python scripts/check.py   # what `make` runs, without make (Windows)
```
