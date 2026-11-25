#!/bin/bash
# Record the FreeBASIC demos as mobile-friendly MP4s.
#
# Usage:
#   tools/record-demo-videos.sh [demo ...]     # no args = all seven
#   LLAMA_CPP=/path/to/llama.cpp tools/record-demo-videos.sh animal_tank rpg_village
#
# Environment:
#   LLAMA_CPP      fork checkout (default: $HOME/Projects/llama.cpp)
#   LLAMA_BUILD_DIR  explicit directory containing libllama.so
#
# Requirements: ffmpeg/ffprobe with x11grab + libx264, xdotool, ImageMagick
# identify (optional but used for the frame colour check).
#
# Rootless XWayland cannot be region-grabbed from the X root: that yields a
# black frame with only the cursor. x11grab's -window_id targets the demo
# window directly, and -draw_mouse 0 keeps the pointer out of the capture.
#
# This script changes no source code and makes no commits. Output files land
# in demo-videos/ and are intentionally left for the owner to place.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLES="$ROOT/examples"
OUT="$ROOT/demo-videos"
LLAMA_CPP="${LLAMA_CPP:-$HOME/Projects/llama.cpp}"
[ -d "$LLAMA_CPP" ] || { echo "llama.cpp checkout not found: $LLAMA_CPP" >&2; exit 1; }

BUILD_DIR="${LLAMA_BUILD_DIR:-}"
if [ -z "$BUILD_DIR" ]; then
    for d in "$LLAMA_CPP/build-cuda/bin" "$LLAMA_CPP/build/bin"; do
        if [ -f "$d/libllama.so" ]; then BUILD_DIR="$d"; break; fi
    done
fi
[ -n "$BUILD_DIR" ] || { echo "no llama.cpp build dir containing libllama.so" >&2; exit 1; }

WORK="$(mktemp -d /tmp/opencode/demo-videos.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

ALL_DEMOS=(animal_tank rpg_village grammar_forge ask_the_codex text_adventure persistent_companion procedural_describer)
if [ $# -eq 0 ]; then WANT=("${ALL_DEMOS[@]}"); else WANT=("$@"); fi

find_window() {
    local title="$1" id name
    for id in $(DISPLAY=:0 xdotool search --name "$title" 2>/dev/null); do
        name="$(DISPLAY=:0 xdotool getwindowname "$id" 2>/dev/null || true)"
        case "$name" in *"$title"*) printf '%s\n' "$id"; return 0;; esac
    done
    return 1
}

# record NAME TITLE MODE TIMEOUT COMMAND...
# MODE direct       - run COMMAND, capture the window titled TITLE
#      xterm-app    - run COMMAND in xterm, capture its fbgfx window titled TITLE
#      xterm-capture- run COMMAND in xterm, capture that xterm window
record() {
    local name="$1" title="$2" mode="$3" timeout="$4"; shift 4
    local -a cmd=("$@")
    local log="$WORK/$name.run.log" raw="$WORK/raw_$name.mp4" fflog="$WORK/$name.ffmpeg.log"
    local xtitle="$name-xterm" app_pid ff_pid win w h
    local ready="$WORK/$name.ready" trigger="$WORK/$name.trigger" done="$WORK/$name.done"
    local helper="$WORK/${name}_xterm.sh" cmdfile="$WORK/${name}_argv.txt"
    local capture_title="$title" launched now hung=0 failed=0

    case "$mode" in
        direct)
            nohup "${cmd[@]}" > "$log" 2>&1 < /dev/null &
            app_pid=$!
            ;;
        xterm-app|xterm-capture)
            [ "$mode" = xterm-capture ] && capture_title="$xtitle"
            rm -f "$ready" "$trigger" "$done"
            : > "$cmdfile"
            local arg
            for arg in "${cmd[@]}"; do printf '%s\n' "$arg" >> "$cmdfile"; done
            cat > "$helper" <<'HELPER'
#!/bin/bash
mapfile -t argv < "$1"
printf '\033]0;%s\007' "$2"
echo READY > "$3"
while [ ! -f "$4" ]; do sleep 0.1; done
cd "$5" || exit 1
"${argv[@]}"
echo DONE > "$6"
sleep 180
HELPER
            chmod +x "$helper"
            nohup xterm -geometry 100x30 -bg black -fg white \
                -fa 'DejaVu Sans Mono:size=13' -T "$xtitle" \
                -e bash "$helper" "$cmdfile" "$xtitle" "$ready" "$trigger" "$EXAMPLES" "$done" \
                > "$log" 2>&1 < /dev/null &
            app_pid=$!
            ;;
        *)
            echo "record: bad mode '$mode'" >&2
            return 2
            ;;
    esac

    launched=$(date +%s)
    win=""
    while [ -z "$win" ]; do
        win="$(find_window "$capture_title" || true)"
        if [ -z "$win" ]; then
            if [ "$mode" = direct ] && ! kill -0 "$app_pid" 2>/dev/null; then
                echo "$name: app exited before its window appeared" >&2
                tail -20 "$log" >&2 || true
                return 3
            fi
            now=$(date +%s)
            if [ $((now - launched)) -ge 120 ]; then
                echo "$name: window '$capture_title' never appeared" >&2
                return 4
            fi
            sleep 0.5
        fi
    done
    eval "$(DISPLAY=:0 xdotool getwindowgeometry --shell "$win")"
    w="$WIDTH"; h="$HEIGHT"
    echo "$name: window=$win '$(DISPLAY=:0 xdotool getwindowname "$win")' ${w}x${h}"

    DISPLAY=:0 ffmpeg -hide_banner -loglevel warning \
        -f x11grab -framerate 30 -draw_mouse 0 -window_id "$win" \
        -video_size "${w}x${h}" -i :0 \
        -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
        -profile:v high -level 4.1 -r 30 -an -movflags +faststart \
        -y "$raw" > "$fflog" 2>&1 &
    ff_pid=$!
    sleep 1
    if ! kill -0 "$ff_pid" 2>/dev/null; then
        echo "$name: ffmpeg failed to start" >&2
        tail -20 "$fflog" >&2 || true
        return 5
    fi

    if [ "$mode" != direct ]; then
        for _ in $(seq 1 100); do [ -f "$ready" ] && break; sleep 0.1; done
        touch "$trigger"
    fi

    while :; do
        if [ "$mode" = direct ]; then
            kill -0 "$app_pid" 2>/dev/null || break
        elif [ -f "$done" ]; then
            break
        fi
        now=$(date +%s)
        if ! kill -0 "$ff_pid" 2>/dev/null; then
            echo "$name: ffmpeg exited early" >&2
            tail -20 "$fflog" >&2 || true
            failed=1
            break
        fi
        if [ $((now - launched)) -ge "$timeout" ]; then
            echo "$name: hung past ${timeout}s; stopping" >&2
            hung=1
            kill -TERM "$app_pid" 2>/dev/null || true
            break
        fi
        sleep 1
    done

    sleep 1
    kill -INT "$ff_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$ff_pid" 2>/dev/null || break; sleep 0.5; done
    kill -KILL "$ff_pid" 2>/dev/null || true
    wait "$ff_pid" 2>/dev/null || true
    [ "$mode" = direct ] || kill -TERM "$app_pid" 2>/dev/null || true

    echo "$name: captured $(( $(date +%s) - launched ))s"
    [ -s "$raw" ] || { echo "$name: no capture output" >&2; tail -20 "$fflog" >&2 || true; return 6; }
    [ "$hung" -eq 0 ] || return 7
    [ "$failed" -eq 0 ] || return 8
    return 0
}

verify() {
    python3 - "$1" "$WORK/verify_$(basename "$1").png" <<'PY'
import json, subprocess, sys
path, png = sys.argv[1:]
j = json.loads(subprocess.check_output([
    'ffprobe', '-v', 'error',
    '-show_entries', 'stream=codec_type,codec_name,pix_fmt,width,height,profile,level',
    '-show_entries', 'format=duration,size', '-of', 'json', path]))
streams = j['streams']
v = next((s for s in streams if s['codec_type'] == 'video'), None)
if v is None:
    sys.exit('no video stream: ' + path)
dur = float(j['format']['duration'])
size = int(j['format']['size'])
subprocess.check_call(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                       '-ss', str(dur / 2), '-i', path, '-frames:v', '1', '-y', png])
try:
    colors = int(subprocess.check_output(['identify', '-format', '%k', png],
                                        stderr=subprocess.DEVNULL))
    solid = colors <= 100
except (OSError, subprocess.CalledProcessError, ValueError):
    colors = -1
    stats = subprocess.check_output(['ffmpeg', '-hide_banner', '-loglevel', 'info',
                                     '-i', png, '-vf', 'signalstats,metadata=print',
                                     '-f', 'null', '-'], stderr=subprocess.STDOUT).decode()
    def stat(key):
        for line in stats.splitlines():
            if 'lavfi.signalstats.' + key + '=' in line:
                return float(line.split('=')[-1])
        return 0.0
    solid = (stat('YMAX') - stat('YMIN')) <= 20
data = open(path, 'rb').read()
faststart = 0 < data.find(b'moov') < data.find(b'mdat')
ok = (v['codec_name'] == 'h264' and v['pix_fmt'] == 'yuv420p'
      and len(streams) == 1 and v.get('profile') == 'High' and v.get('level') == 41
      and v['width'] <= 1280 and v['height'] <= 720 and dur > 0.5
      and faststart and not solid)
print(f"{path}: {v['codec_name']} {v['pix_fmt']} {v['width']}x{v['height']} "
      f"{v.get('profile')}/{v.get('level')} {dur:.2f}s {size/1048576:.2f}MiB "
      f"colors={colors} faststart={faststart} {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
}

for demo in "${WANT[@]}"; do
    case "$demo" in
        animal_tank)
            if record animal_tank 'Animal Tank - Raon-1-nano' direct 300 \
                "$EXAMPLES/run_animal_tank.sh" "$LLAMA_CPP" --demo; then
                cp "$WORK/raw_animal_tank.mp4" "$OUT/demo_animal_tank.mp4"
            else echo "FAILED: animal_tank" >&2; fi ;;
        rpg_village)
            # run_rpg.sh treats argv[2] as a build dir, so pass it explicitly.
            if record rpg_village 'RPG Village - Raon-1-nano' direct 300 \
                "$EXAMPLES/run_rpg.sh" "$LLAMA_CPP" "$BUILD_DIR" --demo; then
                cp "$WORK/raw_rpg_village.mp4" "$OUT/demo_rpg_village.mp4"
            else echo "FAILED: rpg_village" >&2; fi ;;
        grammar_forge)
            # This demo opens its own 800x600 fbgfx window even when launched in xterm.
            if record grammar_forge 'The Grammar Forge' xterm-app 600 \
                "$EXAMPLES/run_grammar_forge.sh" "$LLAMA_CPP" --demo; then
                cp "$WORK/raw_grammar_forge.mp4" "$OUT/demo_grammar_forge.mp4"
            else echo "FAILED: grammar_forge" >&2; fi ;;
        ask_the_codex)
            if record ask_the_codex 'ask_the_codex-xterm' xterm-capture 300 \
                "$EXAMPLES/run_ask_the_codex.sh" "$LLAMA_CPP" --demo; then
                cp "$WORK/raw_ask_the_codex.mp4" "$OUT/demo_ask_the_codex.mp4"
            else echo "FAILED: ask_the_codex" >&2; fi ;;
        text_adventure)
            if record text_adventure 'Locked In at Fumble Labs' direct 300 \
                "$EXAMPLES/run_text_adventure.sh" "$LLAMA_CPP" --demo; then
                cp "$WORK/raw_text_adventure.mp4" "$OUT/demo_text_adventure.mp4"
            else echo "FAILED: text_adventure" >&2; fi ;;
        persistent_companion)
            rm -f "$EXAMPLES/persistent_companion.session" "$EXAMPLES/persistent_companion.notes"
            if record companion_run1 'Pip - survey vessel Cartographer' direct 420 \
                "$EXAMPLES/run_persistent_companion.sh" "$LLAMA_CPP" --demo \
               && record companion_run2 'Pip - survey vessel Cartographer' direct 420 \
                "$EXAMPLES/run_persistent_companion.sh" "$LLAMA_CPP" --demo; then
                list="$WORK/companion_concat.txt"
                printf "file '%s'\nfile '%s'\n" \
                    "$WORK/raw_companion_run1.mp4" "$WORK/raw_companion_run2.mp4" > "$list"
                ffmpeg -hide_banner -loglevel error -f concat -safe 0 -i "$list" \
                    -c copy -movflags +faststart -y "$OUT/demo_persistent_companion.mp4"
            else echo "FAILED: persistent_companion" >&2; fi ;;
        procedural_describer)
            if record procedural_describer 'Procedural Describer - Raon-1-nano' direct 300 \
                "$EXAMPLES/run_procedural_describer.sh" "$LLAMA_CPP" --demo; then
                cp "$WORK/raw_procedural_describer.mp4" "$OUT/demo_procedural_describer.mp4"
            else echo "FAILED: procedural_describer" >&2; fi ;;
        *)
            echo "unknown demo: $demo" >&2; exit 2 ;;
    esac
done

echo "--- verification ---"
failed=0
for demo in "${WANT[@]}"; do
    f="$OUT/demo_$demo.mp4"
    if [ -f "$f" ]; then verify "$f" || failed=1; else echo "MISSING: $f"; failed=1; fi
done
exit "$failed"
