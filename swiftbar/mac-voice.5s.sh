#!/usr/bin/env bash
# <xbar.title>mac-voice</xbar.title>
# <xbar.desc>State + controls for the mac-voice daemon (ElevenLabs voice agent).</xbar.desc>
# <swiftbar.hideAbout>true</swiftbar.hideAbout>
# <swiftbar.hideRunInTerminal>true</swiftbar.hideRunInTerminal>
# <swiftbar.hideLastUpdated>true</swiftbar.hideLastUpdated>
# <swiftbar.hideDisablePlugin>true</swiftbar.hideDisablePlugin>
# <swiftbar.hideSwiftBar>true</swiftbar.hideSwiftBar>
#
# Reads the daemon's status.json (written on every state change) instead of starting
# Python; the daemon also triggers an immediate refresh via swiftbar://refreshplugin.
MV="$HOME/obsidian/42-Git/infra/my-stt-tts/mac-voice"
STATUS="${MAC_VOICE_STATE_DIR:-$HOME/.local/state/mac-voice}/status.json"

field() { /usr/bin/plutil -extract "$1" raw -o - "$STATUS" 2>/dev/null; }

state=down
wake=false
wake_available=false
if [[ -f "$STATUS" ]]; then
  pid=$(field pid)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    state=$(field state)
    wake=$(field wake)
    wake_available=$(field wake_available)
  fi
fi

case "$state" in
  talking)  echo " | sfimage=mic.fill sfcolor=#e5484d" ;;
  starting) echo " | sfimage=ellipsis.circle" ;;
  idle)     if [[ "$wake" == "true" ]]; then echo " | sfimage=ear"; else echo " | sfimage=mic"; fi ;;
  *)        echo " | sfimage=mic.slash sfcolor=#8e8e93" ;;
esac
echo "---"
case "$state" in
  talking|starting)
    echo "Voice on — talking"
    echo "End conversation | bash=$MV param1=-f terminal=false refresh=true" ;;
  idle)
    if [[ "$wake" == "true" ]]; then echo "Listening for the wake word"; else echo "Voice off"; fi
    echo "Start conversation | bash=$MV param1=-n terminal=false refresh=true" ;;
  *)
    echo "Daemon not running"
    echo "Start daemon | bash=/bin/launchctl param1=kickstart param2=gui/$(id -u)/com.albert.mac-voice terminal=false refresh=true" ;;
esac
if [[ "$state" != down && "$wake_available" == "true" ]]; then
  if [[ "$wake" == "true" ]]; then
    echo "Wake word: on — turn off | bash=$MV param1=-w param2=off terminal=false refresh=true"
  else
    echo "Wake word: off — turn on | bash=$MV param1=-w param2=on terminal=false refresh=true"
  fi
fi
echo "Shortcut: hold v, tap o (iTerm2) | disabled=true"
