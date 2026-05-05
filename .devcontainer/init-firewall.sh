#!/usr/bin/env bash
#
# Egress firewall for the pottslab devcontainer.
#
# Threat model: prevent the container from making outbound connections on
# non-web protocols, block all inbound from the network, and keep the default
# policies default-DROP.  Web traffic (HTTP / HTTPS) is allowed broadly so
# that Claude Code's WebFetch tool can reach arbitrary pages.
#
# What is allowed outbound:
#   - DNS (UDP/TCP 53)
#   - SSH (TCP 22)  — for `git clone git@github.com:…` if ever needed
#   - HTTP  (TCP 80)
#   - HTTPS (TCP 443)
#   - The host / docker bridge (so the VS Code server can talk to the host)
#   - A pinned ipset of package-infrastructure domains used by apt/npm/pip/
#     maven (kept even though 80/443 are open, because it makes the rule
#     order explicit and survives if the broad 80/443 rule is ever tightened)
#
# What is still blocked:
#   - All inbound traffic that isn't part of an ESTABLISHED connection
#   - Outbound on any TCP port other than 22/53/80/443
#   - Outbound UDP other than DNS (no QUIC — HTTPS traffic falls back to
#     TCP 443 which is allowed)
#
# Usage (from inside the container):  sudo /usr/local/bin/init-firewall.sh

set -euo pipefail
IFS=$'\n\t'

if [[ "${EUID}" -ne 0 ]]; then
    echo "init-firewall.sh: must be run as root (use sudo)" >&2
    exit 1
fi

echo "[firewall] flushing existing rules..."
iptables -F
iptables -X
iptables -t nat -F || true
iptables -t nat -X || true
iptables -t mangle -F || true
iptables -t mangle -X || true
ipset destroy allowed-domains 2>/dev/null || true

# Allow loopback.
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow DNS (both UDP and TCP; some resolvers fall back to TCP).
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A INPUT  -p udp --sport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A INPUT  -p tcp --sport 53 -m state --state ESTABLISHED -j ACCEPT

# Allow SSH (for git over ssh, if ever needed).
iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT  -p tcp --sport 22 -m state --state ESTABLISHED -j ACCEPT

# Allow the host-side docker bridge so the VS Code server can function.
HOST_IP="$(ip route show default | awk '/default/ {print $3; exit}')"
if [[ -n "${HOST_IP:-}" ]]; then
    HOST_SUBNET="$(echo "$HOST_IP" | awk -F. '{printf "%s.%s.%s.0/24", $1, $2, $3}')"
    echo "[firewall] allowing host subnet $HOST_SUBNET"
    iptables -A INPUT  -s "$HOST_SUBNET" -j ACCEPT
    iptables -A OUTPUT -d "$HOST_SUBNET" -j ACCEPT
fi

# Build the allow-list ipset.
ipset create allowed-domains hash:net

ALLOWED_DOMAINS=(
    "api.anthropic.com"
    "statsig.anthropic.com"
    "sentry.io"
    "registry.npmjs.org"
    "pypi.org"
    "files.pythonhosted.org"
    "github.com"
    "api.github.com"
    "codeload.github.com"
    "objects.githubusercontent.com"
    "raw.githubusercontent.com"
    "deb.debian.org"
    "security.debian.org"
    "deb.nodesource.com"
    # Maven Central — used if/when the Java overhaul gains a build system
    # that pulls dependencies (currently it's plain javac, so unused).
    "repo.maven.apache.org"
    "repo1.maven.org"
)

for domain in "${ALLOWED_DOMAINS[@]}"; do
    echo "[firewall] resolving $domain"
    ips="$(dig +short A "$domain" | grep -E '^[0-9]+(\.[0-9]+){3}$' || true)"
    if [[ -z "$ips" ]]; then
        echo "[firewall] WARNING: could not resolve $domain — skipping"
        continue
    fi
    while IFS= read -r ip; do
        ipset add allowed-domains "$ip" 2>/dev/null || true
    done <<< "$ips"
done

# Default-deny policies.
iptables -P INPUT   DROP
iptables -P FORWARD DROP
iptables -P OUTPUT  DROP

# Re-accept return traffic.
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Accept outbound traffic to the pinned allow-list.
iptables -A OUTPUT -m set --match-set allowed-domains dst -j ACCEPT

# Broad web egress so WebFetch / curl / pip-from-mirror / etc. can reach
# arbitrary hosts.  HTTP+HTTPS only; non-web protocols remain blocked.
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 80  -j ACCEPT

echo "[firewall] rules installed; running smoke tests..."

# Positive smoke test 1: Anthropic API must remain reachable (allow-list).
if ! curl -sS --connect-timeout 5 --max-time 10 https://api.anthropic.com/ -o /dev/null; then
    echo "[firewall] ERROR: api.anthropic.com is not reachable after firewall init" >&2
    exit 2
fi
echo "[firewall] OK: api.anthropic.com reachable"

# Positive smoke test 2: an arbitrary HTTPS host must now be reachable so
# that WebFetch works.  example.com is a good canary — deliberately stable,
# neutral, and always 200-or-404.
if ! curl -sS --connect-timeout 5 --max-time 10 https://example.com/ -o /dev/null; then
    echo "[firewall] ERROR: example.com is not reachable — broad HTTPS egress is broken" >&2
    exit 3
fi
echo "[firewall] OK: example.com reachable (WebFetch should work)"

# Positive smoke test 3: pypi must be reachable so `pip install` works.
if ! curl -sS --connect-timeout 5 --max-time 10 https://pypi.org/simple/ -o /dev/null; then
    echo "[firewall] ERROR: pypi.org is not reachable — pip installs will fail" >&2
    exit 6
fi
echo "[firewall] OK: pypi.org reachable"

# Negative smoke test: a non-web TCP port must still be blocked.  1.1.1.1
# responds to pings and DNS but has nothing listening on 12345, so the
# bash /dev/tcp probe will either connect (firewall is broken) or time out
# (firewall is working).  `timeout 3` caps the wait at 3 seconds.
if timeout 3 bash -c 'exec 3<>/dev/tcp/1.1.1.1/12345' 2>/dev/null; then
    echo "[firewall] ERROR: outbound tcp/12345 to 1.1.1.1 succeeded — firewall is not restricting non-web ports" >&2
    exit 4
fi
echo "[firewall] OK: non-web TCP ports blocked"

# Structural check: the default OUTPUT policy must be DROP.  This is the
# ultimate backstop if any of the allow rules get accidentally reordered.
if ! iptables -S OUTPUT | grep -q '^-P OUTPUT DROP'; then
    echo "[firewall] ERROR: default OUTPUT policy is not DROP" >&2
    exit 5
fi
echo "[firewall] OK: default OUTPUT policy is DROP"

echo "[firewall] initialization complete."
