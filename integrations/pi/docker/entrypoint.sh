#!/bin/sh
set -eu

# A read-only-root launch normally mounts a fresh tmpfs at /tmp, so recreate
# the process-owned state directories on every container start. This also works
# when Docker replaces the image user with the host uid/gid.
umask 077
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" /tmp/.X11-unix

# Debian's xvfb-run waits for a SIGUSR1 readiness handshake that is not
# reliable in every container runtime. Start Xvfb directly, wait for its Unix
# socket, then exec the Agent process so Docker signals (especially SIGINT)
# reach Node without a shell intermediary. TCP listening and X access control
# are unnecessary inside this single-container display server.
display_number=99
Xvfb ":${display_number}" -screen 0 1280x1024x24 -nolisten tcp -ac >/tmp/xvfb.log 2>&1 &
xvfb_pid=$!
attempt=0
while [ ! -S "/tmp/.X11-unix/X${display_number}" ]; do
    if ! kill -0 "${xvfb_pid}" 2>/dev/null; then
        echo "Xvfb failed to start" >&2
        cat /tmp/xvfb.log >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "${attempt}" -ge 100 ]; then
        echo "Timed out waiting for Xvfb" >&2
        cat /tmp/xvfb.log >&2
        exit 1
    fi
    sleep 0.05
done

export DISPLAY=":${display_number}"
exec "$@"
