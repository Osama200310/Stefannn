# Projektarchitektur – Klassenübersicht

Dieses Projekt besteht aus drei Hauptkomponenten zur Bilderfassung und -verarbeitung von Kameras. Hier ist die Struktur kurz erklärt.

---

## 1. **Camera-Klasse** (`camera.py`)

Abstraktionsebene für Kamerahardware. Unterstützt zwei Backends und verwaltet Ressourcen mit Context Manager.

### Hauptattribute
- `use_picam` (bool): Nutzt Picamera2 (Raspberry Pi) oder fällt auf OpenCV zurück
- `picam2`: Picamera2-Instanz (nur wenn auf RPi)
- `cv_cap`: OpenCV VideoCapture-Objekt (Webcam-Fallback)

### Hauptmethoden

| Methode | Beschreibung |
|---------|-------------|
| `__init__(use_picam=True)` | Initialisiert Picamera2 oder fallback auf OpenCV-Webcam |
| `capture_file(filename)` | Speichert ein Bild direkt in eine Datei (JPEG) |
| `capture_array()` | Gibt Bild als BGR-Numpy-Array zurück (für In-Memory-Verarbeitung) |
| `set_controls(controls)` | Setzt Belichtung/ISO/AWB (nur Picamera2; OpenCV ignoriert) |
| `stop()` | Gibt Kamera-Ressourcen frei |
| `__enter__` / `__exit__` | Context Manager für automatisches Cleanup |

### Workflow
```
Camera() → configure → capture_file() / capture_array() → stop()
```

---

## 2. **Detector-Klasse** (`detector.py`)

Führt Objekterkennung mit Ultralytics YOLO durch. Optional: filtert nach bestimmten Klassen und speichert annotierte Bilder.

### Hauptattribute
- `enabled` (bool): Ist Objekterkennung aktiviert?
- `model_path` (str): YOLO-Modellpfad (z.B. `yolov8n.pt`)
- `conf` (float): Confidence-Schwelle (default 0.25)
- `save_annotated` (bool): Speichert Bilder mit bounding boxes (`_det.jpg`)
- `imgsz` (int): Bildgröße für Inferenz (default 640)
- `yolo_classes` (list): Beschränkung auf bestimmte Klassen (IDs/Namen)
- `save_on` (list): Nur speichern, wenn diese Labels erkannt werden
- `_model`: Interne YOLO-Instanz

### Hauptmethoden

| Methode | Beschreibung |
|---------|-------------|
| `__init__(...)` | Lädt YOLO-Modell; normalisiert Filter-Labels (case-insensitive, alphanumerisch) |
| `process_frame(frame)` | Erkennung auf Numpy-Array (BGR/RGB); gibt `(matched, det_lines, annotated)` zurück |
| `process_image(image_path)` | Erkennung auf JPEG-Datei; schreibt `.txt` mit Erkennungen; gibt `(txt_path, matched)` zurück |

### Rückgabewerte

**`process_frame()` returns:**
```
(matched: bool, det_lines: list[str], annotated: ndarray|None, objects: list[str])
  - matched: True wenn ein Label aus save_on erkannt wurde (oder save_on=None)
  - det_lines: Erkennungszeilen im Format "classname score x1 y1 x2 y2"
  - annotated: BGR-Bild mit bounding boxes (wenn save_annotated=True)
  - objects: Liste erkannter Klassennamen
```

**`process_image()` returns:**
```
(txt_path: Path, matched: bool)
  - txt_path: Pfad zur geschriebenen `.txt`-Datei
  - matched: True wenn ein Label aus save_on erkannt wurde
```

### Besonderheiten
- **Label-Normalisierung**: "Ball" → "sportsball", Vergleich case-insensitive und alphanumerisch
- **Zwei Erkennungspfade**:
  - **In-Memory** (`process_frame`): Für schnelle Filterung vor dem Speichern
  - **Datei-basiert** (`process_image`): Für Detailanalyse gespeicherter Bilder
- **Annotationen**: Optional mit bounding boxes und Konfidenz-Score

---

## 3. **Hauptprogramm** (`main.py`)

Orchestriert den Gesamtablauf: Kamera-Aufnahme → Objekterkennung → Speicherung und optionale Callbacks.

### Globale Konfiguration
```python
SAVE_DIR = Path("./timelapse")     # Ausgabeverzeichnis
INTERVAL_SEC = 5                    # Sekunden zwischen Aufnahmen
DETECT_ENABLE = True                # Objekterkennung aktiviert
YOLO_MODEL = "yolov8n.pt"          # Nano-Modell (schnell, klein)
YOLO_CONF = 0.25                   # Confidence-Schwelle
SAVE_ANNOTATED = False              # Annotierte Bilder speichern?
```

### Wichtigste Funktionen

| Funktion | Zweck |
|----------|-------|
| `parse_args()` | CLI-Argumente auslesen (--save-on, --model, --interval, etc.) |
| `apply_exposure_controls(camera, args)` | Setzt Belichtung/ISO (RPi Picamera2 nur) |
| `_run_on_match(...)` | Führt externes Kommando aus, wenn Label erkannt |
| `_extract_labels_from_lines(lines)` | Extrahiert Klassennamen aus Erkennungs-`.txt` |
| `_filter_labels_for_save_on(labels, save_on)` | Filtert Labels nach --save-on Kriterium |
| `_canon(s)` | Normalisiert Label (lowercase, nur alphanumerisch) |
| `main()` | Hauptschleife: Capture → Detect → Save/Filter → Callback |

### Ablauf

```
1. Argumente parsen (CLI-Optionen)
2. Camera und Detector initialisieren
3. Hauptschleife (bis Ctrl+C):
   a) Bild von Kamera erfassen (capture_file oder capture_array)
   b) Optional: In-Memory-Erkennung (schneller Filter)
   c) Nach --save-on filtern (nur speichern wenn Match)
   d) Optional: on-match-Kommando ausführen
   e) Interval-Pause
4. Cleanup und Exit
```

### Zwei Erkennungspfade

#### **Pfad A: Fast Path (mit --save-on)**
```
capture_array() → process_frame() → matched? → save_file() → on-match-cmd()
```
✅ Schnell: Nur speichern wenn erforderliches Label erkannt
✅ I/O-effizient: Keine unnötigen Schreibvorgänge

#### **Pfad B: Default Path (ohne speziellen Filter)**
```
capture_file() → process_image() → on-match-cmd()
```
✅ Speichert alle Bilder (dann optional filtern mit --save-on)
✅ Annotierte Bilder optional mit `--save-annotated`

---

## CLI-Optionen (Auswahl)

```bash
# Kamera-Einstellungen
python3 main.py --save-dir ./output --interval 10
python3 main.py --no-ae --shutter-us 1000 --iso 400

# Objekterkennung
python3 main.py --model yolov8m.pt --conf 0.5 --imgsz 416
python3 main.py --yolo-classes "person" "car" 32

# Filterung & Speicherung
python3 main.py --save-on orange apple person
python3 main.py --save-annotated
python3 main.py --detect / --no-detect

# Callbacks
python3 main.py --on-match-cmd "echo 'Found: {labels}' >> log.txt"
python3 main.py --on-match-sync --on-match-timeout 5
```

---

## Dateiausgabe

Nach einer Aufnahme mit aktivierter Erkennung:

```
timelapse/
├── image_2025-12-20_14-30-45.jpg       # Original-Bild
├── image_2025-12-20_14-30-45.txt       # Erkennungen (classname score x1 y1 x2 y2)
├── image_2025-12-20_14-30-45_det.jpg   # Annotiertes Bild (mit --save-annotated)
└── ...
```

**`.txt` Format Beispiel:**
```
person 0.876 100.5 50.2 300.1 500.8
car 0.654 400.0 150.0 700.0 600.0
```

---

## Zusammenfassung

| Komponente | Aufgabe | Abhängigkeiten |
|------------|---------|-----------------|
| **Camera** | Bildbeschaffung (Hardware-Abstraktiongebung) | picamera2 (opt.) / opencv-python |
| **Detector** | Objekterkennung + Annotation | ultralytics |
| **main.py** | Orchestrierung + CLI + Callbacks | Camera + Detector |

**Workflow in kurz:**
```
CLI-Args → Camera-Init → Loop: { capture → detect → filter → save → callback }
```
