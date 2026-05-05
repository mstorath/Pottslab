#!/usr/bin/env bash
#
# Runs once on container creation (and again after every rebuild).
# Responsibilities:
#   1. Install the egress firewall.
#   2. Install the Python port in editable mode (if a pyproject.toml exists).
#   3. Wire up ~/.claude and ~/.claude.json as symlinks into the persistent
#      named volume at ~/.claude-persist, so the OAuth session, per-project
#      memory, and settings survive container rebuilds.
#
# Idempotent — safe to re-run.

set -euo pipefail

echo "[post-create] installing egress firewall..."
sudo /usr/local/bin/init-firewall.sh

# The Python port doesn't exist yet.  When it does and ships a pyproject.toml
# at the repo root (or under ./python/), this block installs it editable.
if [[ -f "/workspace/pyproject.toml" ]]; then
    echo "[post-create] installing project (root pyproject.toml) in editable mode..."
    pip install --user -e /workspace
elif [[ -f "/workspace/python/pyproject.toml" ]]; then
    echo "[post-create] installing project (python/pyproject.toml) in editable mode..."
    pip install --user -e /workspace/python
else
    echo "[post-create] no pyproject.toml found yet — skipping editable install."
fi

echo "[post-create] wiring Claude Code login persistence..."

PERSIST="$HOME/.claude-persist"
mkdir -p "$PERSIST/.claude"

# Seed ~/.claude.json with the "onboarding complete" flags on first run so
# Claude Code does not fall into its interactive first-run wizard (which
# requires an OAuth browser callback and hangs in devcontainers).  If the
# file already exists we patch it in place — Claude Code occasionally
# rewrites the file and strips these flags, so this is idempotent.
if [[ ! -f "$PERSIST/.claude.json" ]]; then
    printf '{}' > "$PERSIST/.claude.json"
fi
python3 - "$PERSIST/.claude.json" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        data = {}
except (json.JSONDecodeError, FileNotFoundError):
    data = {}
for key in (
    "hasCompletedOnboarding",
    "hasCompletedProjectOnboarding",
    "hasTrustDialogAccepted",
    "bypassPermissionsModeAccepted",
):
    data[key] = True
with open(path, "w") as f:
    json.dump(data, f, indent=2)
PY

# On a rebuild, the container home is wiped but the volume isn't, so we just
# (re-)create the symlinks each time.  `ln -sfn` replaces an existing link.
ln -sfn "$PERSIST/.claude"      "$HOME/.claude"
ln -sfn "$PERSIST/.claude.json" "$HOME/.claude.json"

# ----------------------------------------------------------------------------
# Token check.  `claude /login` inside a devcontainer hangs because its OAuth
# callback listens on a random localhost port that isn't forwarded out of the
# container (anthropics/claude-code#20793).  The reliable path is to generate
# a long-lived token with `claude setup-token` on a machine where OAuth works
# (your macOS host) and export it as $CLAUDE_CODE_OAUTH_TOKEN so devcontainer
# .json can pass it through via remoteEnv.
# ----------------------------------------------------------------------------
if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
    echo "[post-create] CLAUDE_CODE_OAUTH_TOKEN is set — Claude Code will skip /login."
elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[post-create] ANTHROPIC_API_KEY is set — Claude Code will use API-key auth."
else
    cat <<'EOF' >&2
[post-create] WARNING: no CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY found
[post-create]   in the container environment.  `claude /login` will HANG
[post-create]   because the random OAuth callback port is not forwarded out
[post-create]   of the devcontainer.
[post-create]
[post-create]   Fix: on a machine where `claude setup-token` works, run
[post-create]     claude setup-token
[post-create]   and copy the token it prints.  Then on your macOS host (NOT
[post-create]   inside this container), make it visible to GUI VS Code by
[post-create]   ONE of the following:
[post-create]
[post-create]     a) Put it in ~/.zshenv (read by ALL zsh shells, including
[post-create]        the one VS Code spawns):
[post-create]           echo 'export CLAUDE_CODE_OAUTH_TOKEN="sk-..."' >> ~/.zshenv
[post-create]        NB: ~/.zshrc is NOT enough — VS Code launched from the
[post-create]        Dock doesn't source it.
[post-create]
[post-create]     b) Register it with launchd so every GUI app sees it:
[post-create]           launchctl setenv CLAUDE_CODE_OAUTH_TOKEN "sk-..."
[post-create]
[post-create]     c) Always launch VS Code from a terminal via `code .`
[post-create]        after `export`-ing the var in that terminal session.
[post-create]
[post-create]   Then "Dev Containers: Rebuild Container" and this warning
[post-create]   should disappear.
EOF
fi

echo "[post-create] done.  Run 'yolo' to launch Claude Code."
