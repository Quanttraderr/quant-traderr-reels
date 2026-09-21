#!/usr/bin/env python3
"""
insta_post.py
=============
Veröffentlicht ein fertiges Reel direkt auf @quant.traderr, über Metas
Instagram API with Instagram Login, also denselben Zugang, den instaauto schon
benutzt. Kein Browser, keine inoffiziellen Endpunkte.

DER ABLAUF BEI META
    1. Container anlegen   POST /me/media  (media_type=REELS, video_url, caption)
    2. Warten              GET  /<container>?fields=status_code  bis FINISHED
    3. Veröffentlichen     POST /me/media_publish  (creation_id=<container>)

    Meta lädt das Video selbst von der angegebenen URL. Eine lokale Datei kann
    man nicht hochladen, die URL muss also öffentlich erreichbar sein.

WAS DAS SKRIPT NICHT KANN
    Instagrams Musikbibliothek gibt es nur in der App. Was hier rausgeht, trägt
    genau die Tonspur der MP4-Datei und läuft als Original-Audio. Deshalb wird
    der Ton vorher mit musikbett.py untergelegt.

TOKEN
    Aus der Umgebungsvariable IG_TOKEN oder aus einer Datei, die mit --token
    angegeben wird. Das Token wird nie ausgegeben, auch nicht in Fehlern.

AUFRUF
    export IG_TOKEN="..."
    ~/reels/.venv/bin/python ~/reels/insta_post.py --pruefen
    ~/reels/.venv/bin/python ~/reels/insta_post.py --thema MarketMemory \\
        --video-url https://... --posten
"""
import argparse, json, os, sys, time, urllib.error, urllib.parse, urllib.request

REELS = os.path.dirname(os.path.abspath(__file__))
BASE = "https://graph.instagram.com/v23.0"
WARTE_MAX = 300           # Sekunden, die Meta zum Verarbeiten bekommt


def hole_token(pfad=None):
    if pfad:
        with open(os.path.expanduser(pfad)) as fh:
            t = fh.read().strip()
        if t:
            return t
    t = os.environ.get("IG_TOKEN", "").strip()
    if not t:
        raise SystemExit(
            "kein Token. Entweder IG_TOKEN setzen oder --token <datei> angeben.")
    return t


def ruf(pfad, token, params=None, post=False):
    """Ein Aufruf gegen die Graph API. Das Token steht immer im Body oder in
    der Query, taucht aber in keiner Ausgabe auf."""
    p = dict(params or {})
    p["access_token"] = token
    url = f"{BASE}/{pfad}"
    try:
        if post:
            r = urllib.request.urlopen(
                urllib.request.Request(url, data=urllib.parse.urlencode(p).encode()),
                timeout=60)
        else:
            r = urllib.request.urlopen(url + "?" + urllib.parse.urlencode(p),
                                       timeout=60)
        return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        leib = e.read().decode(errors="replace").replace(token, "<TOKEN>")
        raise SystemExit(f"Meta antwortet {e.code} auf {pfad}:\n{leib}")


def pruefen(token):
    """Zwei Fragen: gehört das Token zum richtigen Konto, und darf es posten.

    content_publishing_limit ist der ehrlichste Test für die Berechtigung:
    der Endpunkt existiert nur mit instagram_business_content_publish und
    liefert nebenbei das Tageskontingent."""
    me = ruf("me", token, {"fields": "user_id,username"})
    print(f"Konto: @{me.get('username')}  (id {me.get('user_id')})")
    try:
        lim = ruf("me/content_publishing_limit", token,
                  {"fields": "config,quota_usage"})
        d = (lim.get("data") or [{}])[0]
        print(f"Veroeffentlichen erlaubt. Kontingent benutzt: "
              f"{d.get('quota_usage')} von {(d.get('config') or {}).get('quota_total')}")
        return True
    except SystemExit as e:
        print(str(e))
        print("\n-> Die Berechtigung instagram_business_content_publish fehlt.")
        print("   Im Meta-App-Dashboard ergaenzen und das Konto neu verbinden.")
        return False


def posten(token, thema, video_url, cover_url=None, trocken=False):
    ordner = os.path.join(REELS, thema)
    cap = open(os.path.join(ordner, "caption.txt")).read().strip()
    tags = open(os.path.join(ordner, "hashtags.txt")).read().strip()
    text = f"{cap}\n\n{tags}"
    print(f"Caption {len(text)} Zeichen, Video {video_url}")
    if trocken:
        print("Probelauf, es wird nichts gesendet.")
        return None

    p = {"media_type": "REELS", "video_url": video_url, "caption": text,
         "share_to_feed": "true"}
    if cover_url:
        p["cover_url"] = cover_url
    con = ruf("me/media", token, p, post=True)
    cid = con["id"]
    print("Container", cid)

    t0 = time.time()
    while True:
        st = ruf(cid, token, {"fields": "status_code,status"})
        code = st.get("status_code")
        if code == "FINISHED":
            print("verarbeitet nach", round(time.time() - t0), "s")
            break
        if code in ("ERROR", "EXPIRED"):
            raise SystemExit(f"Meta meldet {code}: {st.get('status')}")
        if time.time() - t0 > WARTE_MAX:
            raise SystemExit(f"nach {WARTE_MAX}s immer noch {code}, abgebrochen")
        time.sleep(5)

    out = ruf("me/media_publish", token, {"creation_id": cid}, post=True)
    mid = out["id"]
    link = ruf(mid, token, {"fields": "permalink"}).get("permalink")
    print("VEROEFFENTLICHT:", mid)
    print("Link:", link)
    return link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", help="Datei mit dem Token")
    ap.add_argument("--pruefen", action="store_true")
    ap.add_argument("--thema")
    ap.add_argument("--video-url")
    ap.add_argument("--cover-url")
    ap.add_argument("--posten", action="store_true")
    a = ap.parse_args()

    token = hole_token(a.token)
    if a.pruefen:
        sys.exit(0 if pruefen(token) else 1)
    if not (a.thema and a.video_url):
        raise SystemExit("--thema und --video-url werden gebraucht")
    posten(token, a.thema, a.video_url, a.cover_url, trocken=not a.posten)


if __name__ == "__main__":
    main()
