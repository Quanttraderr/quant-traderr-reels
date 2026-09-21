#!/usr/bin/env python3
"""
musikbett.py
============
Erzeugt eine Tonspur für ein Reel und legt sie unter das Video.

Warum selbst erzeugt und nicht aus einer Bibliothek: Instagrams lizenzierte
Musik gibt es nur in der App. Was über die API veröffentlicht wird, trägt genau
die Tonspur, die in der MP4-Datei steckt. Alles, was hier entsteht, ist aus
Sinus, Rauschen und Hüllkurven zusammengerechnet, also ohne fremde Rechte und
ohne Sperre gegen automatische Erkennung.

Der Stil ist bewusst schlicht: Kick auf jeder Zählzeit, ein Bass darunter, ein
Arpeggio aus der Moll-Pentatonik, dazu Hi-Hat und ein Riser in den Schluss.
Das soll nicht auffallen, sondern nur verhindern, dass das Reel stumm läuft.

AUFRUF
    ~/reels/.venv/bin/python ~/reels/musikbett.py <Thema>
        -> <Thema>/topic_mit_ton.mp4  (Video unverändert, Ton neu)
    ~/reels/.venv/bin/python ~/reels/musikbett.py <Thema> --nur-ton
        -> <Thema>/bett.wav
"""
import argparse, os, subprocess, sys

import numpy as np

REELS = os.path.dirname(os.path.abspath(__file__))
SR = 48000
BPM = 124.0
DAUER = 11.7


def huellkurve(n, attack, decay):
    """Kurzer Anschlag, exponentieller Abfall, in Sekunden."""
    a = max(int(attack * SR), 1)
    e = np.exp(-np.arange(n) / max(decay * SR, 1.0))
    e[:a] *= np.linspace(0, 1, a)
    return e


def kick(n):
    t = np.arange(n) / SR
    f = 120.0 * np.exp(-t * 26.0) + 46.0          # Tonhöhensturz macht den Punch
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * huellkurve(n, 0.001, 0.10)


def pluck(n, freq):
    t = np.arange(n) / SR
    # Drei Teiltöne statt eines Sägezahns: klingt weicher und rauscht nicht in
    # den Höhen, wenn Instagram das Ganze nochmal durch AAC dreht.
    w = (np.sin(2 * np.pi * freq * t)
         + 0.42 * np.sin(2 * np.pi * 2 * freq * t)
         + 0.18 * np.sin(2 * np.pi * 3 * freq * t))
    return w * huellkurve(n, 0.004, 0.13)


def hat(n):
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, n)
    x = np.diff(np.concatenate([[0.0], x]))       # billiger Hochpass
    return x * huellkurve(n, 0.001, 0.025)


def bett(dauer=DAUER):
    n = int(dauer * SR)
    aus = np.zeros(n)
    beat = 60.0 / BPM
    schritt = beat / 2.0                          # Achtel

    def lege(sig, bei):
        i = int(bei * SR)
        j = min(i + len(sig), n)
        if i < n:
            aus[i:j] += sig[:j - i]

    # A-Moll-Pentatonik, zwei Oktaven, als laufende Figur
    skala = [220.00, 261.63, 329.63, 392.00, 440.00, 523.25]
    figur = [0, 2, 4, 2, 3, 1, 4, 5]

    k = 0
    zeit = 0.0
    while zeit < dauer:
        if k % 2 == 0:
            lege(kick(int(0.30 * SR)) * 0.85, zeit)
        if k % 2 == 1:
            lege(hat(int(0.05 * SR)) * 0.11, zeit)
        # Das Arpeggio setzt erst nach dem Hook ein und wird lauter, damit das
        # Bild und der Ton an derselben Stelle aufmachen.
        stark = min(max((zeit - 1.4) / 4.0, 0.0), 1.0)
        if stark > 0:
            lege(pluck(int(0.36 * SR), skala[figur[k % len(figur)]])
                 * 0.16 * stark, zeit)
        zeit += schritt
        k += 1

    t = np.arange(n) / SR
    # Bass, mit einer Senke auf jeder Zählzeit, damit die Kick durchkommt
    duck = 1.0 - 0.55 * np.exp(-((t % beat) / 0.11))
    aus += 0.30 * np.sin(2 * np.pi * 55.0 * t) * duck
    aus += 0.10 * np.sin(2 * np.pi * 110.0 * t) * duck

    # Riser in die letzten zwei Sekunden
    r0 = int((dauer - 2.6) * SR)
    rng = np.random.default_rng(11)
    rausch = rng.normal(0, 1, n - r0)
    rampe = np.linspace(0, 1, n - r0) ** 2.2
    aus[r0:] += 0.09 * np.cumsum(rausch * rampe) / SR * 40.0

    aus[:int(0.05 * SR)] *= np.linspace(0, 1, int(0.05 * SR))
    aus[-int(0.35 * SR):] *= np.linspace(1, 0, int(0.35 * SR))

    spitze = float(np.abs(aus).max())
    if spitze == 0:
        raise SystemExit("leere Tonspur, das darf nicht passieren")
    aus = aus / spitze * 0.89                     # rund -1 dBFS
    return np.stack([aus, aus], axis=1)           # stereo, identisch


def schreibe_wav(pfad, x):
    import wave
    daten = np.clip(x, -1, 1)
    pcm = (daten * 32767).astype("<i2").tobytes()
    with wave.open(pfad, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("thema")
    ap.add_argument("--nur-ton", action="store_true")
    a = ap.parse_args()

    ordner = os.path.join(REELS, a.thema)
    if not os.path.isdir(ordner):
        raise SystemExit(f"kein Ordner {ordner}")
    wav = os.path.join(ordner, "bett.wav")
    schreibe_wav(wav, bett())
    print("Ton:", wav)
    if a.nur_ton:
        return

    video = os.path.join(ordner, "topic.mp4")
    if not os.path.exists(video):
        raise SystemExit(f"kein Video {video}, erst die Pipeline laufen lassen")
    ziel = os.path.join(ordner, "topic_mit_ton.mp4")
    # Das Bild wird nicht neu kodiert, nur der Ton kommt dazu. Zweimal x264
    # über dieselben Frames kostet genau die Schärfe, für die nativ gerendert
    # wird.
    r = subprocess.run([ffmpeg_exe(), "-y", "-i", video, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-ar", "48000", "-ac", "2", "-shortest",
                        "-movflags", "+faststart", ziel],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-2000:])
    print("Video mit Ton:", ziel)


if __name__ == "__main__":
    main()
