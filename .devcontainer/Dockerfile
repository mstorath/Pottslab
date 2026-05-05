# Devcontainer image for the pottslab overhaul + Python port.
#
# Intended to host Claude Code in "yolo" mode
# (`claude --dangerously-skip-permissions`) inside a sandbox with a
# restrictive egress firewall.  The base image is Python 3.12 (for the
# Python port being built) with Node.js 20 (for `@anthropic-ai/claude-code`)
# and a JDK + ant (for the original Java implementation under Java/) layered
# on top.

FROM python:3.12-bookworm

ARG TZ=Etc/UTC
ENV TZ="$TZ"
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System packages.  `iptables` / `ipset` / `dnsutils` are required by the
# egress firewall; the rest are standard dev conveniences.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg2 less man-db procps sudo unzip \
        build-essential pkg-config \
        zsh fzf jq \
        iptables ipset iproute2 dnsutils aggregate \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Node.js 20 (for Claude Code).
# ---------------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# JDK + Ant for building the original pottslab Java sources.  The repo ships
# an Eclipse-style Java/src layout (no pom.xml / build.gradle) plus a
# pre-built pottslab-standalone.jar, so plain `javac` / `jar` is enough; we
# include `ant` as a convenience for any build script we may add during the
# overhaul.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk-headless ant \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Non-root user.  `claude` is given passwordless sudo access ONLY for the
# handful of commands the firewall init script needs — everything else still
# requires a password (which is not set), so the user is effectively locked
# out of root.
# ---------------------------------------------------------------------------
ARG USERNAME=claude
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/zsh $USERNAME

# Persist shell history across container rebuilds.
RUN mkdir -p /commandhistory \
    && touch /commandhistory/.zsh_history \
    && chown -R $USERNAME:$USERNAME /commandhistory
ENV HISTFILE=/commandhistory/.zsh_history

# Workspace directory — bind-mounted from the host by devcontainer.json.
RUN mkdir -p /workspace && chown $USERNAME:$USERNAME /workspace
WORKDIR /workspace

# Marker env var so Claude Code knows it is running inside a sandboxed
# devcontainer (enables `--dangerously-skip-permissions` without warnings).
ENV DEVCONTAINER=true

# ---------------------------------------------------------------------------
# Firewall init script + limited sudoers entry.
# ---------------------------------------------------------------------------
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod 0755 /usr/local/bin/init-firewall.sh \
    && printf '%s ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh\n' "$USERNAME" \
        > /etc/sudoers.d/$USERNAME-firewall \
    && chmod 0440 /etc/sudoers.d/$USERNAME-firewall

# ---------------------------------------------------------------------------
# Drop to the non-root user for everything from here on.
# ---------------------------------------------------------------------------
USER $USERNAME

# Pre-create the Claude persistence directory *as the claude user* so that
# when the `pottslab-claude-persist-*` named volume first mounts here, it
# inherits claude:claude ownership.  Without this, Docker initializes a
# missing volume path as root-owned and post-create.sh can't write into it.
RUN mkdir -p /home/$USERNAME/.claude-persist

# Pre-install Claude Code so that spinning up a fresh container is instant.
RUN npm config set prefix "/home/$USERNAME/.npm-global" \
    && npm install -g @anthropic-ai/claude-code

ENV PATH="/home/claude/.npm-global/bin:/home/claude/.local/bin:${PATH}"

# Python dependencies for the Python port.  Kept minimal — extend as the
# port grows.  Using --user so the site-packages live under /home/claude
# and survive rebuilds of the workspace bind mount.
RUN pip install --user --no-cache-dir \
        numpy scipy matplotlib pillow \
        pytest hypothesis \
        ruff mypy

# Shell niceties: a `yolo` alias for Claude Code skip-permissions mode.
RUN printf '%s\n' \
        '# --- pottslab devcontainer ---' \
        'export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"' \
        'export PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"' \
        'alias yolo="claude --dangerously-skip-permissions"' \
        'HISTFILE=/commandhistory/.zsh_history' \
        'HISTSIZE=10000' \
        'SAVEHIST=10000' \
        'setopt APPEND_HISTORY INC_APPEND_HISTORY SHARE_HISTORY' \
        >> /home/$USERNAME/.zshrc
