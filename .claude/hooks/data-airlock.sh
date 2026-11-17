#!/bin/bash
# Data airlock for the More than a Hack match data: keep it out of the AI session.
cmd=$(jq -r '.tool_input.command // ""')
deny() { jq -n --arg r "$1" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'; exit 0; }
# Approved pipeline entry points print aggregates only.
if printf '%s' "$cmd" | grep -qE '^[[:space:]]*(make (setup|baselines|train|submission|validate|score|clean)|\.venv/bin/python -m mtah\.)([[:space:]]|$)' \
   && ! printf '%s' "$cmd" | grep -qE '[|;&>]'; then exit 0; fi
if printf '%s' "$cmd" | grep -qiE 'mtah2027-data|MTAH_DATA_DIR|vendor/official/data|BAR-ATH|\.mp4'; then
  first=$(printf '%s' "$cmd" | awk '{print $1}')
  case "$first" in du|ls|stat|mkdir|shasum|mv|unzip|tar) 
    printf '%s' "$cmd" | grep -qE '(unzip[^|]*-p|tar[^|]*-[a-zA-Z]*O)' && deny "MTAH airlock: no extraction to stdout."
    exit 0 ;; esac
  deny "MTAH airlock: match data may only be touched via make targets / python -m mtah.* that print aggregates."
fi
exit 0
