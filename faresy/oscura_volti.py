#!/usr/bin/env python3
"""Copre i soggetti nel video dimostrativo, lasciando intatte le annotazioni.

Il video annotato nasce da quello pulito: stessa scena, stessi fotogrammi,
piu' i riquadri e le scritte disegnati sopra. Da questa coppia si ricavano due
maschere esatte, senza doversi fidare di un rilevatore:

  soggetto   = dove il video PULITO si discosta dallo sfondo (che e' fisso)
  annotazione = dove il video ANNOTATO si discosta da quello pulito

Il risultato e' lo sfondo pulito, il soggetto sostituito da un mosaico, e sopra
le annotazioni ricopiate pixel per pixel: i riquadri, i numeri di traccia e i
tempi restano nitidi anche dove passano sopra il soggetto.

Uso: oscura_volti.py pulito.mp4 annotato.mp4 uscita.mp4 [--campioni cartella]
"""
import sys, pathlib
import cv2
import numpy as np

# Il blocco va misurato sul volto, non sul fotogramma. Il soggetto piu' vicino
# alla telecamera ha il viso largo circa 160px: con blocchi da 16 restava fatto
# di una novantina di quadretti e si riconosceva benissimo. A 48 il viso sta in
# tre blocchi scarsi, che e' quello che serve.
MOSAICO = 48
SOGLIA_SOGGETTO = 16
SOGLIA_SEGNI = 44
MARGINE = 13      # allarga la maschera del soggetto, per non lasciare bordi


def leggi(path):
    cap = cv2.VideoCapture(str(path))
    fotogrammi = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fotogrammi.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fotogrammi:
        raise SystemExit(f"non leggo {path}")
    return fotogrammi, fps


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    pulito_p, annot_p, uscita_p = sys.argv[1:4]
    campioni = None
    if "--campioni" in sys.argv:
        campioni = pathlib.Path(sys.argv[sys.argv.index("--campioni") + 1])
        campioni.mkdir(parents=True, exist_ok=True)

    puliti, fps = leggi(pulito_p)
    annotati, _ = leggi(annot_p)
    if len(puliti) != len(annotati):
        raise SystemExit(f"fotogrammi diversi: {len(puliti)} contro {len(annotati)}")
    H, W = puliti[0].shape[:2]

    # Lo sfondo e' fisso e i soggetti lo attraversano: la mediana nel tempo,
    # pixel per pixel, e' lo sfondo senza nessuno davanti.
    sfondo = np.median(np.stack(puliti), axis=0).astype(np.uint8)

    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MARGINE, MARGINE))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vw = cv2.VideoWriter(uscita_p, fourcc, fps, (W, H))
    if not vw.isOpened():                     # senza H.264 si ripiega su mp4v
        vw = cv2.VideoWriter(uscita_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        print("attenzione: H.264 non disponibile, scrivo in mp4v")

    coperti = 0
    for i, (pul, ann) in enumerate(zip(puliti, annotati)):
        diff_sfondo = cv2.absdiff(pul, sfondo).max(axis=2)
        soggetto = (diff_sfondo > SOGLIA_SOGGETTO).astype(np.uint8)
        soggetto = cv2.morphologyEx(soggetto, cv2.MORPH_CLOSE, nucleo)
        soggetto = cv2.dilate(soggetto, nucleo, iterations=2)

        diff_segni = cv2.absdiff(ann, pul).max(axis=2)
        segni = (diff_segni > SOGLIA_SEGNI).astype(np.uint8)

        piccolo = cv2.resize(pul, (max(1, W // MOSAICO), max(1, H // MOSAICO)),
                             interpolation=cv2.INTER_AREA)
        mosaico = cv2.resize(piccolo, (W, H), interpolation=cv2.INTER_NEAREST)

        fuori = pul.copy()
        m = soggetto.astype(bool)
        fuori[m] = mosaico[m]
        s = segni.astype(bool)
        fuori[s] = ann[s]

        if m.any():
            coperti += 1
        vw.write(fuori)

        if campioni is not None and i in (0, 20, 40, 60, 80, 100, 130, 160, 186):
            cv2.imwrite(str(campioni / f"fot{i:03d}.png"), fuori)

    vw.release()
    print(f"{len(puliti)} fotogrammi scritti, soggetto coperto in {coperti}")


if __name__ == "__main__":
    main()
