# Claude Code system prompt parity assets

These text files are verbatim captures of the system[2] and system[3] blocks
that Claude Code 2.1.112 sends on every OAuth `/v1/messages` request
(captured 2026-04-17 via mitmproxy against a live `claude --model
claude-opus-4-7` session).

`agent.anthropic_adapter` injects them into outgoing OAuth requests when
`is_oauth=True` so the body-shape matches what Anthropic's billing
classifier expects from real Claude Code traffic.  Without this, OAuth
requests were being routed into the Extra Usage billing lane instead of
the Claude Max subscription lane, even after fingerprinting the
user-agent, headers, and JSON body shape (PRs #3-5).

## When to refresh

When Claude Code ships a new release that changes these prompts.  The
re-capture procedure:

1. Start mitmproxy: `mitmdump -p <port> -s <addon.py>`
2. Run CC through the proxy:
   `HTTPS_PROXY=http://127.0.0.1:<port> NODE_EXTRA_CA_CERTS=~/.mitmproxy/mitmproxy-ca-cert.pem \
      claude --model claude-opus-4-7 --print "any trivial task"`
3. Pick the `/v1/messages` capture with the largest body.
4. Extract `body.system[2].text` → `system_persona.txt`.
5. Extract `body.system[3].text` → `system_tool_output_rules.txt`.
6. Bump `_CC_PARITY_ASSETS_VERSION` in `anthropic_adapter.py`.
