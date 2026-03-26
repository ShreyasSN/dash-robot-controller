#!/usr/bin/env python3
"""
Dash Robot Controller (All-in-One): Stable Face Tracking + Vosk ASR + Optional Ollama NLU + Piper TTS + Tablet/Med + BLE
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, asyncio, threading, time, math, json, random, datetime, glob, logging, re, tempfile, subprocess, shutil, platform, wave, contextlib
from pathlib import Path
import numpy as np
import warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dash-app")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

# Optional libs
try:
    import aubio, sounddevice as sd
    AUBIO_AVAILABLE = True
except Exception:
    AUBIO_AVAILABLE = False

try:
    import face_recognition
    FACE_LIB_AVAILABLE = True
except Exception:
    FACE_LIB_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except Exception:
    SPEECH_AVAILABLE = False

try:
    import vosk
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False

try:
    from bleak import BleakScanner
    BLEAK_AVAILABLE = True
except Exception:
    BLEAK_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Robot (if available in your repo)
try:
    from dash.robot import DashRobot, discover_and_connect
except Exception:
    DashRobot = None
    discover_and_connect = None

import cv2
from PIL import Image
import urllib.request as _urlreq
from urllib.error import URLError, HTTPError

# OpenCV contrib (aruco + legacy tracker)
ARUCO_AVAILABLE = False
try:
    from cv2 import aruco as aruco
    ARUCO_AVAILABLE = True
except Exception:
    ARUCO_AVAILABLE = False

LEGACY_AVAILABLE = hasattr(cv2, "legacy")

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QCheckBox, QFrame, QSlider, QColorDialog, QProgressBar, QFileDialog,
    QComboBox, QInputDialog, QMessageBox, QTabWidget, QGridLayout, QGroupBox, QLineEdit
)
from PyQt5.QtGui import QPainter, QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QObject, QUrl, QEasingCurve, QPropertyAnimation, QRect, QEvent
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# ------------ Config ------------
ENABLE_ROBOT_DISCOVERY = True

# Vision
FACE_DETECT_FRAME_STRIDE = 2
TABLET_DETECT_FRAME_STRIDE = 4
TABLET_MIN_AREA = 80
TABLET_MAX_AREA = 1200

# Stabilization/persistence
BOX_PERSIST_MS = 500
TRACKER_REINIT_STRIDE = 12

# Timings / cooldowns
REMINDER_CHECK_INTERVAL_MS = 20_000
REMINDER_WINDOW_MINUTES = 8
GREET_COOLDOWN_SEC = 70
TABLET_VOICE_COOLDOWN_SEC = 35

# Head tracking
HEAD_TRACK_MAX_YAW = 90
HEAD_TRACK_MAX_PITCH_UP = -15
HEAD_TRACK_MAX_PITCH_DOWN = 25
HEAD_TRACK_SMOOTH = 0.12
HEAD_TRACK_DEADZONE_PX = 70
HEAD_TRACK_FACE_LOST_TIMEOUT = 0.9

# Files
FACES_DIR = Path("faces"); FACES_DIR.mkdir(exist_ok=True)
FACE_EMBED_FILE = FACES_DIR / "user_face.npy"
MED_SCHEDULE_FILE = Path("med_schedule.json")
INTENT_FILE = Path("speech_intents.json")
USER_PROFILE_FILE = Path("user_profile.json")
DEFAULT_USER_NAME = "friend"
CHOREO_FILE = Path("choreo_steps.json")

# Tablet modes
TABLET_MODE_COLOR = "Color (basic)"
TABLET_MODE_ARUCO = "ArUco (robust)"
ALLOWED_ARUCO_IDS = {10, 11, 12}

# Ollama (optional local LLM) defaults
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")

def clamp(v,a,b): return max(a,min(b,v))

def hue_to_hex(h: float) -> str:
    h = h % 1.0
    i = int(h * 6); f = (h * 6) - i; q = 1 - f
    if i == 0: r,g,b=1,f,0
    elif i == 1: r,g,b=q,1,0
    elif i == 2: r,g,b=0,1,f
    elif i == 3: r,g,b=0,q,1
    elif i == 4: r,g,b=f,0,1
    else: r,g,b=1,0,q
    return "#{:02X}{:02X}{:02X}".format(int(r*255),int(g*255),int(b*255))

# ------------ Helper UI widgets ------------
class AnimatedButton(QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._anim = QPropertyAnimation(self, b"geometry")
        self._base = None
        self._scale = 0.92
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self._scale_anim(self._scale)
        super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        self._scale_anim(1.0); super().mouseReleaseEvent(e)
    def _scale_anim(self, factor):
        if self._base is None: self._base = self.geometry()
        g = self._base
        nw = int(g.width() * factor); nh = int(g.height() * factor)
        dx = (g.width() - nw) // 2; dy = (g.height() - nh) // 2
        target = QRect(g.x() + dx, g.y() + dy, nw, nh)
        self._anim.stop()
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.OutQuad)
        self._anim.setStartValue(self.geometry())
        self._anim.setEndValue(target)
        self._anim.start()

class JoystickWidget(QFrame):
    def __init__(self, move_callback, knob_color=QColor(255, 215, 0), parent=None):
        super().__init__(parent)
        self.setFixedSize(160, 160)
        self.setStyleSheet("background:#1e1e1e; border-radius:80px;")
        self.center = QPoint(80, 80)
        self.knob_pos = QPoint(80, 80)
        self.knob_radius = 24
        self.dragging = False
        self.move_callback = move_callback
        self.knob_color = knob_color

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(45, 45, 52))
        p.drawEllipse(self.center, 75, 75)
        p.setBrush(QColor(self.knob_color.red(), self.knob_color.green(), self.knob_color.blue(), 220))
        p.drawEllipse(self.knob_pos, self.knob_radius, self.knob_radius)

    def mousePressEvent(self, e):
        if (self.knob_pos - e.pos()).manhattanLength() <= self.knob_radius + 10:
            self.dragging = True

    def mouseMoveEvent(self, e):
        if self.dragging:
            d = e.pos() - self.center
            max_r = 60
            if d.manhattanLength() > max_r:
                d = d * max_r / d.manhattanLength()
            self.knob_pos = self.center + d
            self.update()
            self.emit_move()

    def mouseReleaseEvent(self, e):
        self.dragging = False
        the_center = self.center
        self.knob_pos = the_center
        self.update()
        self.emit_move(reset=True)

    def emit_move(self, reset=False):
        dx = self.knob_pos.x() - 80
        dy = 80 - self.knob_pos.y()
        mv = 60
        nx = dx / mv if not reset else 0
        ny = dy / mv if not reset else 0
        self.move_callback(nx, ny)

class GuiBridge(QObject):
    status_update = pyqtSignal(str)
    sensor_update = pyqtSignal(str)
    enable_buttons = pyqtSignal(bool)
    camera_update = pyqtSignal(QImage)

# ------------ Camera widget ------------
class CameraWidget(QLabel):
    def __init__(self,parent=None,width=640,height=480):
        super().__init__(parent)
        self.setFixedSize(width,height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#000; border:2px solid #333; border-radius:12px;")
        self.setText("Camera not started")
        self._last_frame=None
    def update_frame(self,qimg:QImage):
        self._last_frame=qimg
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def save_snapshot(self):
        if self._last_frame is not None:
            fn=f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            self._last_frame.save(fn,"JPG"); return fn
        return None

# ------------ Stabilizer ------------
class FaceStabilizer:
    def __init__(self):
        self.tracker=None
        self.last_box=None   # (t,r,b,l)
        self.last_box_time=0
        self.frame_idx=0
    def _new_tracker(self):
        # Prefer MOSSE, then KCF; try legacy and modern
        try:
            if LEGACY_AVAILABLE and hasattr(cv2.legacy,'TrackerMOSSE_create'):
                return cv2.legacy.TrackerMOSSE_create()
        except Exception: pass
        try:
            if hasattr(cv2,'TrackerMOSSE_create'):
                return cv2.TrackerMOSSE_create()
        except Exception: pass
        try:
            if LEGACY_AVAILABLE and hasattr(cv2.legacy,'TrackerKCF_create'):
                return cv2.legacy.TrackerKCF_create()
        except Exception: pass
        try:
            if hasattr(cv2,'TrackerKCF_create'):
                return cv2.TrackerKCF_create()
        except Exception: pass
        return None
    def update(self, frame_rgb, detected_boxes):
        h, w, _ = frame_rgb.shape
        self.frame_idx += 1
        now_ms = time.time() * 1000
        chosen = None
        if detected_boxes:
            cx=w//2; cy=h//2
            best, bestd=None,1e18
            for (t,r,b,l) in detected_boxes:
                bx=(l+r)//2; by=(t+b)//2
                d=(bx-cx)**2+(by-cy)**2
                if d<bestd: best,bestd=(t,r,b,l),d
            chosen=best
            self.last_box=chosen; self.last_box_time=now_ms
            if self.tracker is None or (self.frame_idx % TRACKER_REINIT_STRIDE)==0:
                self.tracker=self._new_tracker()
                if self.tracker is not None:
                    x,y=self.last_box[3],self.last_box[0]
                    bw,bh=self.last_box[1]-self.last_box[3], self.last_box[2]-self.last_box[0]
                    try: self.tracker.init(cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR),(x,y,bw,bh))
                    except Exception: self.tracker=None
        else:
            if self.tracker is not None:
                try:
                    ok,rect=self.tracker.update(cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR))
                except Exception:
                    ok=False; rect=(0,0,0,0)
                if ok:
                    x,y,bw,bh=map(int,rect)
                    t,r,b,l=y,x+bw,y+bh,x
                    chosen=(t,r,b,l)
                    self.last_box=chosen
                else:
                    self.tracker=None
        if chosen is None and self.last_box is not None:
            if (time.time()*1000 - self.last_box_time) <= BOX_PERSIST_MS:
                chosen=self.last_box
            else:
                self.last_box=None
        return chosen

# ------------ Face Manager ------------
class FaceRecognitionManager:
    def __init__(self):
        self.enabled=False
        self.last_greet_time=0
        self.user_encoding=None
        self.haar=None
        self.match_threshold=0.55
        self._load_enc()
        # Setup detectors
        try:
            hp=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
            if os.path.exists(hp): self.haar=cv2.CascadeClassifier(hp)
        except Exception:
            self.haar=None
    def _load_enc(self):
        if FACE_EMBED_FILE.exists():
            try: self.user_encoding=np.load(str(FACE_EMBED_FILE))
            except Exception: self.user_encoding=None
    def detect_faces(self, frame_rgb):
        boxes=[]
        if FACE_LIB_AVAILABLE:
            try:
                small=cv2.resize(frame_rgb,(0,0),fx=0.5,fy=0.5)
                locs=face_recognition.face_locations(small, model="hog")
                boxes=[(t*2,r*2,b*2,l*2) for (t,r,b,l) in locs]
            except Exception: boxes=[]
        if not boxes and self.haar is not None:
            try:
                gray=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2GRAY)
                faces=self.haar.detectMultiScale(gray,1.15,6)
                for (x,y,w,h) in faces: boxes.append((y,x+w,y+h,x))
            except Exception:
                pass
        return boxes
    def recognize_user(self, frame_rgb, box):
        if not (FACE_LIB_AVAILABLE and self.user_encoding is not None and box):
            return False
        try:
            encs=face_recognition.face_encodings(frame_rgb,[box])
            if encs: return np.linalg.norm(encs[0]-self.user_encoding)<self.match_threshold
        except Exception: pass
        return False
    def should_greet(self):
        now=time.time()
        if now-self.last_greet_time>GREET_COOLDOWN_SEC:
            self.last_greet_time=now; return True
        return False

# ------------ Medication Manager ------------
class MedicationManager:
    def __init__(self):
        self.schedule={}
        self.last_reminded={}
        self.enabled=True
        self.next_upcoming=""
        self.load()
    def load(self):
        if MED_SCHEDULE_FILE.exists():
            try:
                with open(MED_SCHEDULE_FILE,"r") as f: self.schedule=json.load(f)
            except Exception: self.schedule={}
    def save(self):
        try:
            with open(MED_SCHEDULE_FILE,"w") as f: json.dump(self.schedule,f,indent=2)
        except Exception: pass
    def _parse_12h_time(self, s):
        s = s.strip().lower().replace(".", "")
        m = re.match(r'^(\d{1,2})(?::?(\d{2}))?\s*(am|pm)$', s)
        if not m: return None
        hh = int(m.group(1)); mm = int(m.group(2) or 0); ap=m.group(3)
        if hh==12: hh=0
        if ap=='pm': hh+=12
        if 0<=hh<24 and 0<=mm<60:
            return f"{hh:02d}:{mm:02d}"
        return None
    def add_medicine(self,name,times):
        cleaned=[]
        for t in times:
            t=t.strip()
            m=re.match(r'^(\d{1,2}):(\d{2})$', t)
            if m:
                hh,mm=int(m.group(1)),int(m.group(2))
                if 0<=hh<24 and 0<=mm<60: cleaned.append(f"{hh:02d}:{mm:02d}")
                continue
            t12=self._parse_12h_time(t)
            if t12: cleaned.append(t12)
        if cleaned:
            self.schedule[name]=sorted(set(self.schedule.get(name,[])+cleaned))
            self.save(); return True, self.schedule[name]
        return False,[]
    def due_now(self):
        if not self.enabled: return []
        now=datetime.datetime.now()
        out=[]
        for name,times in self.schedule.items():
            for t in times:
                target=datetime.datetime.combine(now.date(), datetime.time(int(t[:2]),int(t[3:])))
                diff=abs((now-target).total_seconds())/60.0
                if diff<=REMINDER_WINDOW_MINUTES:
                    key=(name,t,now.date().isoformat())
                    if not self.last_reminded.get(key): out.append((name,t,diff))
        return out
    def mark_reminded(self,name,t):
        key=(name,t,datetime.date.today().isoformat())
        self.last_reminded[key]=True
    def ack_first_due(self):
        due=self.due_now()
        if not due: return False,None
        name,t,_=due[0]; self.mark_reminded(name,t); return True,(name,t)
    def compute_next_upcoming(self):
        now=datetime.datetime.now()
        soonest=None; sn=""
        for name,times in self.schedule.items():
            for t in times:
                target=datetime.datetime.combine(now.date(), datetime.time(int(t[:2]),int(t[3:])))
                if target<now: target+=datetime.timedelta(days=1)
                if soonest is None or target<soonest:
                    soonest=target; sn=name
        if soonest:
            delta=soonest-now
            hrs,rem=divmod(int(delta.total_seconds()),3600); mins=rem//60
            self.next_upcoming=f"{sn} in {hrs}h {mins}m"
        else: self.next_upcoming=""

# ------------ Piper TTS (robust, no text/bytes mismatch) ------------
class PiperTTS:
    def __init__(self, piper_bin=None, model_path=None, length_scale=1.15, noise_scale=0.55):
        self.piper_bin = piper_bin or os.environ.get("PIPER_BIN", "piper")
        self.model_path = model_path or os.environ.get("PIPER_MODEL", "")
        self.length_scale = str(length_scale)  # kept for compatibility
        self.noise_scale = str(noise_scale)    # kept for compatibility
        self.last_error = None
        if "ESPEAK_NG_DATA_PATH" not in os.environ:
            cand = Path(__file__).resolve().parent / "piper" / "piper" / "espeak-ng-data"
            if cand.exists():
                os.environ["ESPEAK_NG_DATA_PATH"] = str(cand)

    def available(self) -> bool:
        exe_ok = False
        try:
            if self.piper_bin and (os.path.isfile(self.piper_bin) or shutil.which(self.piper_bin)):
                exe_ok = True
        except Exception:
            exe_ok = False
        model_ok = bool(self.model_path and os.path.isfile(self.model_path))
        return exe_ok and model_ok

    def _run_piper(self, args, text_bytes, cwd=None):
        self.last_error = None
        try:
            p = subprocess.run(
                [self.piper_bin] + args,
                input=text_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                cwd=cwd,
                check=False,
            )
            stdout = p.stdout.decode("utf-8", errors="ignore") if p.stdout else ""
            stderr = p.stderr.decode("utf-8", errors="ignore") if p.stderr else ""
            return p.returncode, stdout, stderr
        except Exception as e:
            self.last_error = f"exec error: {e}"
            return None, "", ""

    def synth_to_wav(self, text: str):
        """Synthesize text to a temporary WAV file using Piper. Returns path or None."""
        self.last_error = None
        if not self.available():
            self.last_error = "Piper not available (bin or model missing)"
            return None
        text_bytes = text.encode("utf-8") if isinstance(text, str) else (text or b"")
        work_dir = tempfile.mkdtemp(prefix="dash_piper_work_")
        out_wav = str(Path(work_dir) / f"piper_{int(time.time()*1000)}.wav")
        args = ["-m", self.model_path, "-q", "-w", out_wav]
        rc, out, err = self._run_piper(args, text_bytes, cwd=work_dir)
        if rc is None:
            self.last_error = f"piper run failed: {err or out}"
            return None
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 44:
            return out_wav
        self.last_error = f"piper rc={rc}, out={out.strip()[:160]}, err={err.strip()[:160]}"
        return None

# ------------ LLM NLU via Ollama (optional) ------------
def _http_get_json(url: str, timeout: float = 2.5):
    try:
        with _urlreq.urlopen(url, timeout=timeout) as resp:
            import json as _json
            return _json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None

def ollama_chat(text: str, url: str, model: str, timeout: float = 12.0):
    import json, urllib.request
    SYSTEM = "You are a concise, friendly robot assistant. Respond conversationally in one or two short sentences."
    try:
        req = urllib.request.Request(
            f"{url}/api/chat",
            data=json.dumps({
                "model": model,
                "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":text}],
                "stream": False
            }).encode("utf-8"),
            headers={"Content-Type":"application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message") or {}).get("content","").strip()
    except Exception:
        return ""

def nlu_ollama_parse(text: str, url: str, model: str, timeout: float = 8.0):
    import json, urllib.request
    SYSTEM = """You are a voice control NLU for a small robot.
Return only JSON: {"actions":[ ... ]} and at most one {"type":"reply","text":"..."} as a conversational answer.
If unclear, return {"actions":[{"type":"noop"}]}.

Actions:
- {"type":"set_name","name":"string"} 
- {"type":"notify","text":"string"}
- {"type":"reply","text":"string"}
- {"type":"move","direction":"forward|backward","speed":int(Optional)}
- {"type":"turn","direction":"left|right","speed":int(Optional)}
- {"type":"stop"}
- {"type":"head","yaw":int(Optional),"pitch":int(Optional)}
- {"type":"head_track","enabled":true|false}
- {"type":"lights","mode":"set","color":"#RRGGBB"}
- {"type":"lights","mode":"party","enabled":true|false}
- {"type":"dance","enabled":true|false}
- {"type":"choreo_start"}; {"type":"choreo_stop"}
- {"type":"choreo_add_step","step":"spin_left|spin_right|blink|sway|color:#RRGGBB|head:yaw:INT|head:pitch:INT"}
- {"type":"choreo_clear"}
- {"type":"music_mode","enabled":true|false}
- {"type":"choreo","name":"Classic|Extreme|Robot Wheels|Party Lights"}
- {"type":"teach_steps"}
- {"type":"face_register"}
- {"type":"camera","action":"on|off|capture|record_on|record_off"}
- {"type":"tablet_detect","enabled":true|false}
- {"type":"tablet_mode","mode":"ArUco|Color"}
- {"type":"tablet_query"}
- {"type":"med_add","name":"string","times":["HH:MM or 12h like 5 pm", "..."]}
- {"type":"med_reminders","enabled":true|false}
- {"type":"med_mark_taken"}
- {"type":"med_query_due"}
- {"type":"med_show_schedule"}

Name-setting phrases like "call me Rishi", "my name is Ana", "I'm Ravi" should produce only set_name."""
    try:
        req = urllib.request.Request(
            f"{url}/api/chat",
            data=json.dumps({
                "model": model,
                "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":text}],
                "stream": False
            }).encode("utf-8"),
            headers={"Content-Type":"application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data.get("message") or {}).get("content","")
        try:
            obj=json.loads(content)
            if isinstance(obj,dict) and "actions" in obj: return obj
        except Exception:
            pass
        m=re.search(r'\{.*\}', content, flags=re.S)
        if m:
            try:
                obj=json.loads(m.group(0))
                if isinstance(obj,dict) and "actions" in obj: return obj
            except Exception: pass
        return {"actions":[{"type":"noop"}]}
    except Exception:
        return {"actions":[{"type":"noop"}]}

# ------------ Main Controller ------------
class DashFullController(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dash Robot Controller (All-in-One)")
        self.resize(1380, 900)

        # Fade-in (keep a strong reference to avoid GC)
        self.setWindowOpacity(0.0)
        self._fade = QPropertyAnimation(self, b"windowOpacity")
        self._fade.setDuration(600)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
        self._fade.start()

        # Async loop thread
        self.loop=asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()

        # Robot & managers
        self.robot=None
        self._connecting=False
        self.face_manager=FaceRecognitionManager()
        self.med_manager=MedicationManager()
        self.stab=FaceStabilizer()

        # User/intents
        self.intents=self._load_intents()
        self.user_profile=self._load_user_profile()
        self.user_name=self.user_profile.get("name", DEFAULT_USER_NAME)

        # Camera
        self._last_rgb_frame=None
        self.camera_active=False
        self.camera_stop_event=threading.Event()
        self._camera_thread_lock=threading.Lock()
        self.selected_camera_source=None
        self._cam_pending={"fourcc":"YUYV","width":640,"height":480,"fps":30}
        self._frame_counter=0
        self.tablet_detect_enabled=False
        self.tablet_mode=TABLET_MODE_ARUCO if ARUCO_AVAILABLE else TABLET_MODE_COLOR
        self.last_tablet_voice_time=0
        self.last_tablet_id=None  # e.g., 10, 11, ...

        # Head tracking
        self.head_tracking_enabled=False
        self._head_last_seen=time.time()
        self._head_current_yaw=0.0
        self._head_current_pitch=0.0
        self._head_target_yaw=0.0
        self._head_target_pitch=0.0
        self._move_active=False
        self._deadzone_px=HEAD_TRACK_DEADZONE_PX
        self._deadzone_hyst_px=12
        self._last_sent_yaw=None
        self._last_sent_pitch=None
        self._last_head_send=0.0
        self._min_head_send_dt=0.08

        # Motion
        self.target_left=0.0
        self.target_right=0.0
        self.smooth_task_started=False
        self.raw_wheels_enabled=False
        self._last_raw_send=0
        self._raw_send_interval=0.030
        self._current_speed=260
        self._move_sensitivity=1.0
        self._turn_sensitivity=1.0

        # Speech & NLU/TTS
        self.speech_enabled=False
        self.speech_thread=None
        self.speech_stop_event=threading.Event()
        self.tts_engine=None
        self.piper = PiperTTS()
        self.use_piper_tts = True
        # Auto-enable pyttsx3 fallback when Piper missing
        self.disable_pyttsx3_fallback = True
        if TTS_AVAILABLE and not self.piper.available():
            self.disable_pyttsx3_fallback = False

        self.asr_backend="Vosk" if VOSK_AVAILABLE else "Google"
        self.vosk_model_path=os.environ.get("VOSK_MODEL_PATH","")
        # Optional LLM NLU
        self.use_llm_nlu = True
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_MODEL
        # NLU mode selector: True = Ollama only, False = Rule-based only
        self.force_ollama_only = False
        # Personalized name usage
        self.use_personal_name = True
        # Google SR language + Vosk mic gain
        self.google_language = "en-IN"
        self.mic_gain = 1.4  # default 1.4x

        # Music / Choreography
        self.music_mode = False
        self.beat_thread=None
        self.beat_stop_event=threading.Event()
        self.choreo_steps=self._load_choreo()
        self.custom_choreo=[]
        self._choreo_idx=0

        # UI + Setup
        self._build_gui()
        self._setup_signals()
        we_setup_timers=getattr(self,'_setup_timers',None)
        if we_setup_timers: we_setup_timers()
        else:
            self.sensor_timer=QTimer(); self.sensor_timer.timeout.connect(self.update_sensors); self.sensor_timer.start(500)
            self.reminder_timer=QTimer(); self.reminder_timer.timeout.connect(self.check_reminders); self.reminder_timer.start(REMINDER_CHECK_INTERVAL_MS)
            self.check_reminders()
        self.apply_theme("Dark")

        # Audio player for Qt
        self.audio_player=QMediaPlayer()
        self.light_color="#FFAA00"
        self._auto_react=False

        # Speaking guard to pause ASR during playback
        self._tts_playing = threading.Event()

        # Auto-start camera
        QTimer.singleShot(700, lambda: self.btn_camera.setChecked(True))

        # Auto connect
        if ENABLE_ROBOT_DISCOVERY and discover_and_connect:
            def _submit():
                asyncio.run_coroutine_threadsafe(self.connect_robot(), self.loop)
            self.loop.call_soon_threadsafe(_submit)
        else:
            self.status_label.setText("Robot discovery disabled.")

        # Head tracking updater
        self.head_track_timer=QTimer()
        self.head_track_timer.timeout.connect(self._head_tracking_update)
        self.head_track_timer.start(70)

        # Ollama status updater
        self.ollama_status_timer=QTimer()
        self.ollama_status_timer.timeout.connect(self.update_ollama_status)
        self.ollama_status_timer.start(15000)
        QTimer.singleShot(1200, self.update_ollama_status)

    # ---------- Intents/Profile I/O ----------
    def _default_intents(self):
        return {
            "dance_triggers":["dance","dance for music","start dancing"],
            "compliments":["good job","nice job","awesome","great","love you","good robot","amazing","cool robot"]
        }
    def _default_user_profile(self):
        return {"name": DEFAULT_USER_NAME}
    def _load_intents(self):
        if INTENT_FILE.exists():
            try:
                with open(INTENT_FILE,"r") as f: return json.load(f)
            except Exception: pass
        return self._default_intents()
    def _save_intents(self,data):
        try:
            with open(INTENT_FILE,"w") as f: json.dump(data,f,indent=2)
        except Exception: pass
    def _load_user_profile(self):
        if USER_PROFILE_FILE.exists():
            try:
                with open(USER_PROFILE_FILE,"r") as f: return json.load(f)
            except Exception: pass
        return self._default_user_profile()
    def _save_user_profile(self,data):
        try:
            with open(USER_PROFILE_FILE,"w") as f: json.dump(data,f,indent=2)
        except Exception: pass

    # ---------- GUI ----------
    def _build_gui(self):
        font = QFont(); font.setPointSize(10); self.setFont(font)
        main_layout = QVBoxLayout(self)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Initializing...")
        self.sensor_label = QLabel("Sensors: --")
        self.sensor_label.setStyleSheet("color:#88e1a0;")
        status_layout.addWidget(self.status_label, 2)
        status_layout.addWidget(self.sensor_label, 1)
        main_layout.addLayout(status_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self._create_main_controls_tab()
        self._create_camera_vision_tab()
        self._create_settings_tab()

        prox_layout = QHBoxLayout()
        self.prox_left_bar = QProgressBar(); self.prox_left_bar.setRange(0, 255); self.prox_left_bar.setFormat("Prox L %v")
        self.prox_right_bar = QProgressBar(); self.prox_right_bar.setRange(0, 255); self.prox_right_bar.setFormat("Prox R %v")
        prox_layout.addWidget(self.prox_left_bar)
        prox_layout.addWidget(self.prox_right_bar)
        main_layout.addLayout(prox_layout)
        self.setLayout(main_layout)

    def _create_main_controls_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left_panel = QVBoxLayout()
        self.camera_widget = CameraWidget()
        left_panel.addWidget(self.camera_widget)

        joystick_layout = QHBoxLayout()
        joystick_layout.addStretch()
        self.left_joystick = JoystickWidget(self.on_left_joystick_move, knob_color=QColor(0,200,255))
        self.right_joystick = JoystickWidget(self.on_right_joystick_move, knob_color=QColor(255,130,210))
        joystick_layout.addWidget(self.left_joystick)
        joystick_layout.addWidget(self.right_joystick)
        joystick_layout.addStretch()
        left_panel.addLayout(joystick_layout)
        layout.addLayout(left_panel, 2)

        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignTop)

        head_group = QGroupBox("Head Controls")
        head_layout = QVBoxLayout()
        self.head_yaw_label = QLabel("Head Yaw: 0°")
        self.head_yaw_slider = QSlider(Qt.Horizontal); self.head_yaw_slider.setRange(-90,90); self.head_yaw_slider.setValue(0); self.head_yaw_slider.valueChanged.connect(self.on_head_yaw_change)
        self.head_pitch_label = QLabel("Head Pitch: 0°")
        self.head_pitch_slider = QSlider(Qt.Horizontal); self.head_pitch_slider.setRange(-20,30); self.head_pitch_slider.setValue(0); self.head_pitch_slider.valueChanged.connect(self.on_head_pitch_change)
        self.btn_head_center = AnimatedButton("Center Head"); self.btn_head_center.clicked.connect(self.center_head)
        head_layout.addWidget(self.head_yaw_label)
        head_layout.addWidget(self.head_yaw_slider)
        head_layout.addWidget(self.head_pitch_label)
        head_layout.addWidget(self.head_pitch_slider)
        head_layout.addWidget(self.btn_head_center)
        head_group.setLayout(head_layout)
        right_panel.addWidget(head_group)

        lights_group = QGroupBox("Lights")
        lights_layout = QHBoxLayout()
        self.btn_color = AnimatedButton("Pick Color"); self.btn_color.clicked.connect(self.pick_color)
        self.btn_set_light = AnimatedButton("Set Light"); self.btn_set_light.clicked.connect(self.set_light_color)
        lights_layout.addWidget(self.btn_color)
        lights_layout.addWidget(self.btn_set_light)
        lights_group.setLayout(lights_layout)
        right_panel.addWidget(lights_group)

        actions_group = QGroupBox("Actions & BLE")
        actions_layout = QGridLayout()
        self.btn_hi = AnimatedButton("Hi"); self.btn_hi.clicked.connect(lambda: self.say("hi"))
        self.btn_wave = AnimatedButton("Wave"); self.btn_wave.clicked.connect(lambda: self.send_cmd(self.wave))
        self.btn_spin = AnimatedButton("Spin"); self.btn_spin.clicked.connect(lambda: self.send_cmd(self.spin_preset))
        self.btn_tada = AnimatedButton("Tada"); self.btn_tada.clicked.connect(lambda: self.say("tada"))
        self.btn_rainbow = AnimatedButton("Rainbow"); self.btn_rainbow.clicked.connect(lambda: self.send_cmd(self.led_rainbow))
        self.btn_blink = AnimatedButton("Blink"); self.btn_blink.clicked.connect(lambda: self.send_cmd(self.led_blink))
        self.btn_connect_robot = AnimatedButton("Connect Robot"); self.btn_connect_robot.clicked.connect(self.manual_connect)
        self.btn_ble_scan = AnimatedButton("Scan BLE"); self.btn_ble_scan.clicked.connect(self.scan_ble_once)
        actions_layout.addWidget(self.btn_hi,0,0)
        actions_layout.addWidget(self.btn_wave,0,1)
        actions_layout.addWidget(self.btn_spin,1,0)
        actions_layout.addWidget(self.btn_tada,1,1)
        actions_layout.addWidget(self.btn_rainbow,2,0)
        actions_layout.addWidget(self.btn_blink,2,1)
        actions_layout.addWidget(self.btn_connect_robot,3,0)
        actions_layout.addWidget(self.btn_ble_scan,3,1)
        actions_group.setLayout(actions_layout)
        right_panel.addWidget(actions_group)

        right_panel.addStretch()
        layout.addLayout(right_panel, 1)
        self.tabs.addTab(tab, "Main Controls")

    def _create_camera_vision_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignTop)

        cam_settings_group = QGroupBox("Camera Settings")
        cam_settings_layout = QGridLayout()
        self.camera_select = QComboBox()
        self.btn_refresh_cams = AnimatedButton("Refresh"); self.btn_refresh_cams.clicked.connect(self.refresh_cameras)
        self.btn_use_cam = AnimatedButton("Use"); self.btn_use_cam.clicked.connect(self.use_selected_camera)
        self.cam_format_combo = QComboBox(); self.cam_format_combo.addItems(["YUYV","MJPG"])
        self.cam_res_combo = QComboBox()
        for r in ["640x480@30","640x360@30","432x240@30","320x240@30","1280x720@7.5","1280x960@7.5"]:
            self.cam_res_combo.addItem(r)
        self.btn_apply_cam = AnimatedButton("Apply Cam"); self.btn_apply_cam.clicked.connect(self.apply_camera_settings)
        cam_settings_layout.addWidget(QLabel("Device:"),0,0)
        cam_settings_layout.addWidget(self.camera_select,0,1,1,2)
        cam_settings_layout.addWidget(self.btn_refresh_cams,0,3)
        cam_settings_layout.addWidget(self.btn_use_cam,0,4)
        cam_settings_layout.addWidget(QLabel("Format:"),1,0)
        cam_settings_layout.addWidget(self.cam_format_combo,1,1)
        cam_settings_layout.addWidget(QLabel("Resolution:"),1,2)
        cam_settings_layout.addWidget(self.cam_res_combo,1,3)
        cam_settings_layout.addWidget(self.btn_apply_cam,1,4)
        cam_settings_group.setLayout(cam_settings_layout)
        layout.addWidget(cam_settings_group)
        self.refresh_cameras()

        cam_actions_group = QGroupBox("Camera Actions")
        cam_actions_layout = QHBoxLayout()
        self.btn_camera = AnimatedButton("Start Camera"); self.btn_camera.setCheckable(True); self.btn_camera.toggled.connect(self.on_camera_toggle)
        self.btn_capture = AnimatedButton("Capture"); self.btn_capture.clicked.connect(self.capture_image)
        self.btn_record = AnimatedButton("Record"); self.btn_record.setCheckable(True); self.btn_record.toggled.connect(self.toggle_video_recording)
        cam_actions_layout.addWidget(self.btn_camera)
        cam_actions_layout.addWidget(self.btn_capture)
        cam_actions_layout.addWidget(self.btn_record)
        cam_actions_group.setLayout(cam_actions_layout)
        layout.addWidget(cam_actions_group)

        vision_group = QGroupBox("Vision Features")
        vision_layout = QGridLayout()
        self.face_recog_checkbox = QCheckBox("Face Recognition"); self.face_recog_checkbox.stateChanged.connect(self.on_face_recog_toggle)
        self.head_track_checkbox = QCheckBox("Head Tracking"); self.head_track_checkbox.stateChanged.connect(self.on_head_track_toggle)
        self.tablet_detect_checkbox = QCheckBox("Tablet Detection"); self.tablet_detect_checkbox.stateChanged.connect(self.on_tablet_detect_toggle)
        self.btn_register_face = AnimatedButton("Register My Face"); self.btn_register_face.setEnabled(False); self.btn_register_face.clicked.connect(self.on_register_face)
        self.face_status_label = QLabel("Face: --")
        self.tablet_status_label = QLabel("Tablet: --")
        vision_layout.addWidget(self.face_recog_checkbox,0,0)
        vision_layout.addWidget(self.head_track_checkbox,1,0)
        vision_layout.addWidget(self.tablet_detect_checkbox,2,0)
        vision_layout.addWidget(self.btn_register_face,0,1)
        vision_layout.addWidget(self.face_status_label,1,1)
        vision_layout.addWidget(self.tablet_status_label,2,1)
        vision_group.setLayout(vision_layout)
        layout.addWidget(vision_group)

        tablet_mode_group = QGroupBox("Tablet Detection Mode")
        tablet_mode_layout = QHBoxLayout()
        self.tablet_mode_combo = QComboBox()
        modes = [TABLET_MODE_COLOR]
        if ARUCO_AVAILABLE: modes.insert(0, TABLET_MODE_ARUCO)
        self.tablet_mode_combo.addItems(modes)
        self.tablet_mode_combo.setCurrentText(TABLET_MODE_ARUCO if ARUCO_AVAILABLE else TABLET_MODE_COLOR)
        self.tablet_mode_combo.currentTextChanged.connect(lambda t: setattr(self, "tablet_mode", t))
        tablet_mode_layout.addWidget(QLabel("Mode:"))
        tablet_mode_layout.addWidget(self.tablet_mode_combo)
        if ARUCO_AVAILABLE:
            tablet_mode_layout.addWidget(QLabel(f"Allowed IDs: {sorted(list(ALLOWED_ARUCO_IDS))}"))
        else:
            tablet_mode_layout.addWidget(QLabel("(Install opencv-contrib to enable ArUco)"))
        tablet_mode_group.setLayout(tablet_mode_layout)
        layout.addWidget(tablet_mode_group)

        layout.addStretch()
        self.tabs.addTab(tab,"Camera & Vision")

    def _create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignTop)

        # Movement Sensitivity
        sens_group = QGroupBox("Movement Sensitivity")
        sens_layout = QGridLayout()
        self.left_sens_label = QLabel("Move Sensitivity: 1.00")
        self.left_sens_slider = QSlider(Qt.Horizontal)
        self.left_sens_slider.setRange(10, 100)
        self.left_sens_slider.setValue(50)
        self.left_sens_slider.valueChanged.connect(self.on_left_sens_change)

        self.right_sens_label = QLabel("Turn Sensitivity: 1.00")
        self.right_sens_slider = QSlider(Qt.Horizontal)
        self.right_sens_slider.setRange(10, 100)
        self.right_sens_slider.setValue(50)
        self.right_sens_slider.valueChanged.connect(self.on_right_sens_change)

        sens_layout.addWidget(self.left_sens_label, 0, 0)
        sens_layout.addWidget(self.left_sens_slider, 0, 1)
        sens_layout.addWidget(self.right_sens_label, 1, 0)
        sens_layout.addWidget(self.right_sens_slider, 1, 1)
        sens_group.setLayout(sens_layout)
        layout.addWidget(sens_group)

        # General Settings
        gen_group = QGroupBox("General Settings")
        gen_layout = QGridLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "iOS Light", "iOS Dark", "AMOLED", "High Contrast"])
        self.theme_combo.currentTextChanged.connect(self.apply_theme)

        self.speech_checkbox = QCheckBox("Enable Speech")
        self.speech_checkbox.stateChanged.connect(self.on_speech_toggle)

        self.raw_wheels_checkbox = QCheckBox("Raw Wheels Mode")
        self.raw_wheels_checkbox.stateChanged.connect(self.on_raw_wheels_toggle)

        self.audio_on_pc_checkbox = QCheckBox("Play Sounds on PC")
        self.audio_on_pc_checkbox.setChecked(True)

        self.asr_backend_combo = QComboBox()
        backends = ["Google"]
        if VOSK_AVAILABLE:
            backends.insert(0, "Vosk")
        self.asr_backend_combo.addItems(backends)
        self.asr_backend_combo.setCurrentText("Vosk" if VOSK_AVAILABLE else "Google")
        self.asr_backend_combo.currentTextChanged.connect(self.on_asr_backend_change)

        self.vosk_path_edit = QLineEdit(os.environ.get("VOSK_MODEL_PATH", ""))
        self.vosk_path_edit.setPlaceholderText("Path to Vosk model folder")
        self.vosk_path_edit.editingFinished.connect(self.on_vosk_path_change)

        # Personalized name usage
        self.personal_name_checkbox = QCheckBox("Use your name in responses")
        self.personal_name_checkbox.setChecked(True)
        self.personal_name_checkbox.setToolTip("When ON, the robot will address you by your saved name.")
        self.personal_name_checkbox.stateChanged.connect(
            lambda s: setattr(self, "use_personal_name", s == Qt.Checked)
        )

        # Ollama-only NLU toggle
        self.ollama_only_checkbox = QCheckBox("Use Ollama NLU only")
        self.ollama_only_checkbox.setChecked(False)
        self.ollama_only_checkbox.setToolTip("When ON: use Ollama only. When OFF: use rule-based only.")
        self.ollama_only_checkbox.stateChanged.connect(self.on_force_ollama_toggle)

        # Google language code
        self.google_lang_edit = QLineEdit(self.google_language)
        self.google_lang_edit.setPlaceholderText("Google language (e.g., en-IN, en-GB, en-US)")
        self.google_lang_edit.editingFinished.connect(
            lambda: setattr(self, "google_language", self.google_lang_edit.text().strip() or "en-US")
        )

        # Vosk mic gain
        self.mic_gain_label = QLabel("Mic Gain (Vosk): 1.40x")
        self.mic_gain_slider = QSlider(Qt.Horizontal)
        self.mic_gain_slider.setRange(50, 200)  # 0.5x..2.0x
        self.mic_gain_slider.setValue(140)      # default 1.4x

        def on_mic_gain_change(v):
            self.mic_gain = v / 100.0
            self.mic_gain_label.setText(f"Mic Gain (Vosk): {self.mic_gain:.2f}x")

        self.mic_gain_slider.valueChanged.connect(on_mic_gain_change)

        # Ollama status row
        self.ollama_status_label = QLabel("Ollama: checking...")
        self.ollama_status_label.setStyleSheet("color:#aaa;")
        self.btn_ollama_refresh = AnimatedButton("Refresh Ollama")
        self.btn_ollama_refresh.clicked.connect(lambda: self.update_ollama_status(manual=True))

        # Music mode
        self.music_mode_checkbox = QCheckBox("Dance to music (beat-detect)")
        self.music_mode_checkbox.setChecked(False)
        self.music_mode_checkbox.setToolTip("Requires mic and aubio. Robot will do simple steps on the beat.")
        self.music_mode_checkbox.stateChanged.connect(self.on_music_mode_toggle)

        self.speed_label = QLabel("Base Speed: 260")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(100, 600)
        self.speed_slider.setValue(260)
        self.speed_slider.valueChanged.connect(self.on_speed_change)

        # Place rows
        row = 0
        gen_layout.addWidget(QLabel("UI Theme:"), row, 0)
        gen_layout.addWidget(self.theme_combo, row, 1)
        row += 1

        gen_layout.addWidget(self.speech_checkbox, row, 0)
        gen_layout.addWidget(self.raw_wheels_checkbox, row, 1)
        row += 1

        gen_layout.addWidget(self.audio_on_pc_checkbox, row, 0)
        gen_layout.addWidget(self.personal_name_checkbox, row, 1)
        row += 1

        gen_layout.addWidget(QLabel("ASR Backend:"), row, 0)
        gen_layout.addWidget(self.asr_backend_combo, row, 1)
        row += 1

        gen_layout.addWidget(QLabel("Vosk Model Path:"), row, 0)
        gen_layout.addWidget(self.vosk_path_edit, row, 1)
        row += 1

        gen_layout.addWidget(QLabel("Google Language:"), row, 0)
        gen_layout.addWidget(self.google_lang_edit, row, 1)
        row += 1

        gen_layout.addWidget(self.mic_gain_label, row, 0)
        gen_layout.addWidget(self.mic_gain_slider, row, 1)
        row += 1

        gen_layout.addWidget(self.music_mode_checkbox, row, 0, 1, 2)
        row += 1

        gen_layout.addWidget(self.ollama_status_label, row, 0)
        gen_layout.addWidget(self.btn_ollama_refresh, row, 1)
        row += 1

        gen_layout.addWidget(self.speed_label, row, 0)
        gen_layout.addWidget(self.speed_slider, row, 1)
        row += 1

        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)

        # Medication Management
        med_group = QGroupBox("Medication Management")
        med_layout = QVBoxLayout()
        self.reminders_checkbox = QCheckBox("Enable Reminders")
        self.reminders_checkbox.setChecked(True)
        self.reminders_checkbox.stateChanged.connect(self.on_reminders_toggle)
        self.med_status_label = QLabel("Med: --")

        med_btn_layout = QHBoxLayout()
        self.btn_add_medicine = AnimatedButton("Add Medicine")
        self.btn_add_medicine.clicked.connect(self.on_add_medicine)
        self.btn_show_schedule = AnimatedButton("Show Schedule")
        self.btn_show_schedule.clicked.connect(self.on_show_schedule)
        med_btn_layout.addWidget(self.btn_add_medicine)
        med_btn_layout.addWidget(self.btn_show_schedule)

        med_layout.addWidget(self.reminders_checkbox)
        med_layout.addWidget(self.med_status_label)
        med_layout.addLayout(med_btn_layout)
        med_group.setLayout(med_layout)
        layout.addWidget(med_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Settings & More")

    def apply_theme(self, theme):
        base_bg="#141414"; base_fg="#e0e0e0"; panel="#1f1f24"; border="#2d2d32"; accent="#4DA3FF"
        if theme=="iOS Light": base_bg="#f2f2f7"; base_fg="#222"; panel="#ffffff"; border="#d0d0d5"; accent="#007AFF"
        elif theme=="iOS Dark": base_bg="#1c1c1e"; base_fg="#f5f5f7"; panel="#2c2c2e"; border="#3a3a3c"; accent="#0A84FF"
        elif theme=="AMOLED": base_bg="#000000"; base_fg="#e8e8e8"; panel="#101010"; border="#181818"; accent="#33B5FF"
        elif theme=="High Contrast": base_bg="#000000"; base_fg="#FFFFFF"; panel="#1a1a1a"; border="#FFFFFF"; accent="#FFD500"
        style=f"""
        QWidget {{ background:{base_bg}; color:{base_fg}; font-family:'Segoe UI','Helvetica Neue',Arial; }}
        QTabWidget::pane {{ border-top: 1px solid {border}; }}
        QTabBar::tab {{ background:{panel}; border: 1px solid {border}; border-bottom-color: {border}; padding: 8px 20px; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        QTabBar::tab:selected {{ background: {base_bg}; border-bottom-color: {base_bg}; }}
        QGroupBox {{ background-color: {panel}; border: 1px solid {border}; border-radius: 8px; margin-top: 10px; }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; }}
        QLabel {{ color:{base_fg}; }}
        QPushButton {{ background:{panel}; border:1px solid {border}; border-radius:9px; padding:6px 12px; color:{base_fg}; }}
        QPushButton:hover {{ border:1px solid {accent}; }}
        QPushButton:checked {{ background:{accent}; color:#fff; }}
        QCheckBox::indicator {{ width:18px; height:18px; }}
        QCheckBox {{ spacing:6px; }}
        QSlider::groove:horizontal {{ height:6px; background:{border}; border-radius:3px; }}
        QSlider::handle:horizontal {{ background:{accent}; width:16px; margin:-6px 0; border-radius:8px; }}
        QSlider::groove:vertical {{ width:6px; background:{border}; border-radius:3px; }}
        QSlider::handle:vertical {{ background:{accent}; height:16px; margin:0 -6px; border-radius:8px; }}
        QComboBox {{ background:{panel}; border:1px solid {border}; border-radius:8px; padding:4px 8px; }}
        QComboBox:hover {{ border:1px solid {accent}; }}
        QProgressBar {{ background:{panel}; border:1px solid {border}; border-radius:8px; text-align:center; }}
        QProgressBar::chunk {{ background:{accent}; border-radius:8px; }}
        """
        self.setStyleSheet(style)

    # ---------- Signals/Timers ----------
    def _setup_signals(self):
        self.gui_bridge=GuiBridge()
        self.gui_bridge.status_update.connect(self.status_label.setText)
        self.gui_bridge.sensor_update.connect(self.sensor_label.setText)
        self.gui_bridge.camera_update.connect(self.camera_widget.update_frame)
        self.gui_bridge.enable_buttons.connect(self.set_buttons_enabled)
    def _setup_timers(self):
        self.sensor_timer=QTimer(); self.sensor_timer.timeout.connect(self.update_sensors); self.sensor_timer.start(500)
        self.reminder_timer=QTimer(); self.reminder_timer.timeout.connect(self.check_reminders); self.reminder_timer.start(REMINDER_CHECK_INTERVAL_MS)
        self.check_reminders()

    # ---------- Async loop ----------
    def _run_loop(self):
        asyncio.set_event_loop(self.loop); self.loop.run_forever()

    async def connect_robot(self):
        if self._connecting: return
        if not discover_and_connect:
            self.gui_bridge.status_update.emit("dash.robot not available."); return
        self._connecting=True
        self.gui_bridge.status_update.emit("Connecting to Dash...")
        log.info("Starting BLE discovery/connect")
        try:
            robot=await discover_and_connect()
            if robot and isinstance(robot, DashRobot):
                self.robot=robot
                self.gui_bridge.status_update.emit(f"Connected: {getattr(robot,'address','?')}")
                self.gui_bridge.enable_buttons.emit(True)
                log.info("Connected")
            else:
                self.gui_bridge.status_update.emit("Robot not found.")
                log.warning("discover_and_connect() returned None")
        except Exception as e:
            self.gui_bridge.status_update.emit(f"Conn error: {e}")
            log.exception("Connection failed")
        finally:
            self._connecting=False

    def manual_connect(self):
        if self.robot:
            self.status_label.setText("Already connected."); return
        self.status_label.setText("Re-attempting connection...")
        def _submit(): asyncio.run_coroutine_threadsafe(self.connect_robot(), self.loop)
        self.loop.call_soon_threadsafe(_submit)

    def scan_ble_once(self):
        if not BLEAK_AVAILABLE:
            self.status_label.setText("bleak not installed."); return
        async def _scan():
            try:
                self.gui_bridge.status_update.emit("Scanning BLE (10s)...")
                devs = await BleakScanner.discover(timeout=10.0)
                lines = [f"{d.name or '?'} ({d.address}) RSSI={getattr(d,'rssi',None)}" for d in devs]
                print("[BLE] Found devices:\n  " + "\n  ".join(lines))
                self.gui_bridge.status_update.emit(f"BLE devices found: {len(devs)}")
            except Exception as e:
                self.gui_bridge.status_update.emit(f"BLE scan error: {e}")
                log.exception("BLE scan error")
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_scan()))

    def send_cmd(self, coro_func):
        if self.robot: self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro_func()))

    def set_buttons_enabled(self, enabled):
        btns = [getattr(self,'btn_tada',None), getattr(self,'btn_hi',None), getattr(self,'btn_wave',None),
                getattr(self,'btn_spin',None), getattr(self,'btn_rainbow',None), getattr(self,'btn_blink',None)]
        for b in btns:
            if b: b.setEnabled(enabled)

    # ---------- Camera ----------
    def refresh_cameras(self):
        self.camera_select.clear()
        for p in sorted(glob.glob("/dev/video[0-9]*")): self.camera_select.addItem(p)
        for i in range(6): self.camera_select.addItem(f"Index {i}")
        if self.camera_select.count()==0: self.camera_select.addItem("No devices found")
        self.status_label.setText("Cameras refreshed.")

    def use_selected_camera(self):
        txt=self.camera_select.currentText()
        if txt.startswith("/dev/video"): self.selected_camera_source=txt
        elif txt.startswith("Index"):
            try: self.selected_camera_source=int(txt.split()[-1])
            except Exception: self.selected_camera_source=0
        else: self.selected_camera_source=0
        if self.camera_active:
            self.stop_camera(); self.btn_camera.setChecked(False)
        self.btn_camera.setChecked(True)
        self.status_label.setText(f"Using camera: {self.selected_camera_source}")

    def apply_camera_settings(self):
        fmt=self.cam_format_combo.currentText(); res=self.cam_res_combo.currentText()
        try:
            wh,fps=res.split("@"); w,h=map(int,wh.split("x")); fps=float(fps)
        except Exception:
            w,h,fps=640,480,30
        self._cam_pending={"fourcc":fmt,"width":w,"height":h,"fps":fps}
        self.status_label.setText(f"Cam cfg: {fmt} {w}x{h}@{fps}")
        if self.camera_active:
            self.stop_camera(); self.btn_camera.setChecked(False); self.btn_camera.setChecked(True)

    def on_camera_toggle(self, checked):
        if checked: self.btn_camera.setText("Stop Camera"); self.start_camera()
        else: self.btn_camera.setText("Start Camera"); self.stop_camera()

    def start_camera(self):
        with self._camera_thread_lock:
            if self.camera_active: return
            self.camera_stop_event.clear()
            self.camera_thread=threading.Thread(target=self.camera_loop,daemon=True)
            self.camera_active=True; self.camera_thread.start()

    def stop_camera(self):
        with self._camera_thread_lock:
            if not self.camera_active: return
            self.camera_stop_event.set()
        if getattr(self,"camera_thread",None): self.camera_thread.join(timeout=2)
        self.camera_active=False
        self.gui_bridge.camera_update.emit(self._text_image("Camera stopped"))
        self.btn_register_face.setEnabled(False)

    def camera_loop(self):
        device=self.selected_camera_source
        if device is None:
            device="/dev/video0" if os.path.exists("/dev/video0") else 0
        cfg=self._cam_pending
        try:
            cap=cv2.VideoCapture(device, cv2.CAP_V4L2)
            time.sleep(0.15)
            if not cap.isOpened():
                try_indexes = [0,1,2,3] if not isinstance(device,int) else [device,0,1,2,3]
                for idx in try_indexes:
                    alt = cv2.VideoCapture(idx)
                    time.sleep(0.1)
                    if alt.isOpened():
                        cap = alt
                        device = idx
                        break
            if not cap.isOpened():
                self.gui_bridge.camera_update.emit(self._text_image("Open failed"))
                self.status_label.setText("Camera open failed. Try another Index in Camera Settings.")
                self.camera_active=False; return

            req_fc=cv2.VideoWriter_fourcc(*cfg["fourcc"])
            cap.set(cv2.CAP_PROP_FOURCC, req_fc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
            cap.set(cv2.CAP_PROP_FPS, cfg["fps"]); time.sleep(0.05)

            first_frame=False
            detector_stride_count=0

            while not self.camera_stop_event.is_set():
                ok,frame=cap.read()
                if not ok or frame is None: continue
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue
                frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                self._last_rgb_frame=frame_rgb
                self._frame_counter+=1

                if getattr(self, "recording", False) and getattr(self, "video_writer", None) is not None:
                    try:
                        out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        self.video_writer.write(out_bgr)
                    except Exception:
                        pass

                detector_stride_count += 1
                boxes=[]
                if (self.head_tracking_enabled or self.face_manager.enabled) and (detector_stride_count % FACE_DETECT_FRAME_STRIDE==0):
                    try: boxes=self.face_manager.detect_faces(frame_rgb)
                    except Exception: boxes=[]

                stable_box = self.stab.update(frame_rgb, boxes)

                label_text="Face: --"
                recognized=False
                if stable_box:
                    color=(255,255,0)
                    if self.face_manager.enabled and self.face_manager.user_encoding is not None and FACE_LIB_AVAILABLE:
                        try: recognized=self.face_manager.recognize_user(frame_rgb, stable_box)
                        except Exception: recognized=False
                    if recognized:
                        color=(0,255,0); label_text=f"Face: {self.user_name}"
                        if self.face_manager.should_greet():
                            if self.use_personal_name: self.voice_say(f"hi {self.user_name}")
                            else: self.voice_say("hi friend")
                    else:
                        label_text="Face: Detected"
                    t,r,b,l = stable_box
                    cv2.rectangle(frame_rgb,(l,t),(r,b),color,2)
                    cv2.putText(frame_rgb, self.user_name if recognized else "Unknown",
                                (l, max(0,t-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    self._head_last_seen=time.time()

                if stable_box and self.head_tracking_enabled:
                    self._update_head_track_targets(stable_box, frame_rgb.shape[1], frame_rgb.shape[0])

                if self.tablet_detect_enabled and (self._frame_counter % TABLET_DETECT_FRAME_STRIDE==0):
                    try:
                        detected, tid = self.detect_tablet(frame_rgb)
                    except Exception:
                        detected, tid = False, None
                    self.last_tablet_id = tid if detected else None
                    if detected:
                        self.tablet_status_label.setText(f"Tablet: ID {tid}" if tid is not None else "Tablet: Detected")
                        self.maybe_tablet_voice(tid)
                    else:
                        self.tablet_status_label.setText("Tablet: --")

                self.face_status_label.setText(label_text)

                try:
                    h,w,ch=frame_rgb.shape
                    qimg=QImage(frame_rgb.data,w,h,ch*w,QImage.Format_RGB888)
                    self.gui_bridge.camera_update.emit(qimg)
                except Exception:
                    pass

                if not first_frame:
                    self.btn_register_face.setEnabled(True)
                    first_frame=True

            cap.release()
        except Exception as e:
            log.exception("Camera loop error: %s", e)
        finally:
            self.camera_active=False

    def detect_tablet(self, frame_rgb):
        if self.tablet_mode == TABLET_MODE_ARUCO and ARUCO_AVAILABLE:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            tid = None
            try:
                parameters = aruco.DetectorParameters()
                detector = aruco.ArucoDetector(dictionary, parameters)
                corners, ids, _ = detector.detectMarkers(gray)
            except Exception:
                parameters = aruco.DetectorParameters_create()
                corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
            if ids is not None and len(ids) > 0:
                for idv in ids.flatten():
                    if int(idv) in ALLOWED_ARUCO_IDS:
                        tid = int(idv)
                        break
                return (tid is not None), tid
            return False, None
        else:
            hsv=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            mask=cv2.inRange(hsv,(0,0,185),(180,40,255))
            mask=cv2.medianBlur(mask,5)
            contours,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area=cv2.contourArea(c)
                if TABLET_MIN_AREA < area < TABLET_MAX_AREA:
                    x,y,w,h=cv2.boundingRect(c)
                    aspect=max(w,h)/(min(w,h)+1e-5)
                    if aspect<2.0:
                        return True, None
            return False, None

    def _text_image(self,text):
        img=QImage(self.camera_widget.width(), self.camera_widget.height(), QImage.Format_RGB32)
        img.fill(QColor("black"))
        p=QPainter(img); p.setPen(QColor("white")); p.drawText(img.rect(),Qt.AlignCenter,text); p.end()
        return img

    def capture_image(self):
        fn=self.camera_widget.save_snapshot()
        if fn: self.status_label.setText(f"Captured: {fn}")
        else: self.status_label.setText("No frame to capture. Start camera first.")

    # ---------- Head Tracking ----------
    def on_head_track_toggle(self,state):
        self.head_tracking_enabled=(state==Qt.Checked)
        self.status_label.setText(f"Head Track {'ON' if self.head_tracking_enabled else 'OFF'}")
        if not self.head_tracking_enabled: self.center_head()

    def _update_head_track_targets(self, loc, frame_w, frame_h):
        t,r,b,l=loc
        cx=(l+r)/2.0; cy=(t+b)/2.0
        dx=cx - (frame_w/2.0); dy=cy - (frame_h/2.0)
        rad=math.hypot(dx,dy)
        enter=self._deadzone_px + self._deadzone_hyst_px
        exit_=max(0,self._deadzone_px - self._deadzone_hyst_px)
        if not self._move_active and rad<=enter: return
        if self._move_active and rad<=exit_:
            self._move_active=False; return
        self._move_active=True
        norm_x=dx/(frame_w/2.0); norm_y=dy/(frame_h/2.0)
        target_yaw=int(norm_x * HEAD_TRACK_MAX_YAW * 0.70)
        target_pitch=int(norm_y*(HEAD_TRACK_MAX_PITCH_DOWN if norm_y>0 else -HEAD_TRACK_MAX_PITCH_UP))
        target_pitch=clamp(target_pitch, HEAD_TRACK_MAX_PITCH_UP, HEAD_TRACK_MAX_PITCH_DOWN)
        self._head_target_yaw=clamp(target_yaw, -HEAD_TRACK_MAX_YAW, HEAD_TRACK_MAX_YAW)
        self._head_target_pitch=target_pitch
        self._head_last_seen=time.time()

    def _head_tracking_update(self):
        if not self.robot or not self.head_tracking_enabled: return
        face_visible = (time.time() - self._head_last_seen) < HEAD_TRACK_FACE_LOST_TIMEOUT
        if not face_visible: return
        alpha = HEAD_TRACK_SMOOTH
        self._head_current_yaw = (1-alpha)*self._head_current_yaw + alpha*self._head_target_yaw
        self._head_current_pitch = (1-alpha)*self._head_current_pitch + alpha*self._head_target_pitch
        now_mono=time.monotonic()
        yaw_i=int(round(self._head_current_yaw))
        pitch_i=int(round(self._head_current_pitch))
        should_send=( (self._last_sent_yaw is None) or (abs(yaw_i-self._last_sent_yaw)>=2) or
                      (self._last_sent_pitch is None) or (abs(pitch_i-self._last_sent_pitch)>=2) )
        if should_send and (now_mono - self._last_head_send) >= self._min_head_send_dt:
            self._last_head_send=now_mono
            self._last_sent_yaw=yaw_i; self._last_sent_pitch=pitch_i
            async def go(yaw,pitch):
                try:
                    if hasattr(self.robot,"head_yaw"): await self.robot.head_yaw(yaw)
                    if hasattr(self.robot,"head_pitch"): await self.robot.head_pitch(pitch)
                except Exception: pass
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(go(yaw_i,pitch_i)))

    # ---------- Movement ----------
    def on_left_joystick_move(self,nx,ny):
        forward=clamp(ny*self._move_sensitivity,-1,1)
        steer=clamp(nx*self._turn_sensitivity,-1,1)
        left=clamp(forward+steer*0.6,-1,1)
        right=clamp(forward-steer*0.6,-1,1)
        if self.raw_wheels_enabled:
            self.send_raw_wheel_command(left,right)
        else:
            self.target_left=left; self.target_right=right
            self.ensure_smooth_task()
    def on_right_joystick_move(self,nx,ny):
        if self.raw_wheels_enabled:
            turn=clamp(nx*self._turn_sensitivity,-1,1)
            self.send_raw_wheel_command(turn,-turn)
        else:
            bias=nx*0.3*self._turn_sensitivity
            self.target_left=clamp(self.target_left+bias,-1,1)
            self.target_right=clamp(self.target_right-bias,-1,1)
            self.ensure_smooth_task()
    def ensure_smooth_task(self):
        if self.raw_wheels_enabled: return
        if not self.smooth_task_started:
            self.smooth_task_started=True
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._smooth_wheel_loop()))
    async def _smooth_wheel_loop(self):
        cur_l=0.0; cur_r=0.0; ramp=0.25
        while True:
            await asyncio.sleep(0.04)
            if self.raw_wheels_enabled or not self.robot: continue
            tl,tr=self.target_left,self.target_right
            cur_l+=max(min(tl-cur_l,ramp),-ramp)
            cur_r+=max(min(tr-cur_r,ramp),-ramp)
            ls=int(cur_l*self._current_speed); rs=int(cur_r*self._current_speed)
            try:
                if hasattr(self.robot,"set_wheel_speeds"): await self.robot.set_wheel_speeds(ls,rs)
                elif hasattr(self.robot,"set_wheels"): await self.robot.set_wheels(ls,rs)
                elif hasattr(self.robot,"motors"): await self.robot.motors(ls,rs)
                else:
                    forward=(ls+rs)//2; turn=(ls-rs)//2
                    if abs(turn)>40 and abs(forward)<40: await self.robot.spin(turn)
                    else: await self.robot.drive(forward)
            except Exception: pass

    def on_speed_change(self,v):
        self._current_speed=v; self.speed_label.setText(f"Base Speed: {v}")
    def on_left_sens_change(self,v):
        self._move_sensitivity=v/50; self.left_sens_label.setText(f"Move Sensitivity: {self._move_sensitivity:.2f}")
    def on_right_sens_change(self,v):
        self._turn_sensitivity=v/50; self.right_sens_label.setText(f"Turn Sensitivity: {self._turn_sensitivity:.2f}")
    def on_raw_wheels_toggle(self,state):
        self.raw_wheels_enabled=(state==Qt.Checked)
        self.status_label.setText(f"Raw wheels {'ON' if self.raw_wheels_enabled else 'OFF'}")

    def send_raw_wheel_command(self,ln,rn):
        if not self.robot: return
        now=time.time()
        if now-self._last_raw_send<self._raw_send_interval: return
        self._last_raw_send=now
        max_sp=self._current_speed
        ls=int(ln*max_sp); rs=int(rn*max_sp)
        async def go():
            try:
                if hasattr(self.robot,"set_wheels"): await self.robot.set_wheels(ls,rs)
                elif hasattr(self.robot,"set_wheel_speeds"): await self.robot.set_wheel_speeds(ls,rs)
                elif hasattr(self.robot,"motors"): await self.robot.motors(ls,rs)
                else:
                    forward=(ls+rs)//2; turn=(ls-rs)//2
                    if abs(turn)>40 and abs(forward)<40: await self.robot.spin(turn)
                    else: await self.robot.drive(forward)
            except Exception: pass
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(go()))

    # ---------- Head Manual ----------
    def on_head_yaw_change(self,v):
        self.head_yaw_label.setText(f"Head Yaw: {v}°")
        self._head_target_yaw = v
        if not self.robot or self.head_tracking_enabled: return
        async def go():
            try: await self.robot.head_yaw(v)
            except Exception: pass
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(go()))
    def on_head_pitch_change(self,v):
        self.head_pitch_label.setText(f"Head Pitch: {v}°")
        self._head_target_pitch = v
        if not self.robot or self.head_tracking_enabled: return
        async def go():
            try: await self.robot.head_pitch(v)
            except Exception: pass
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(go()))
    def center_head(self):
        self.head_yaw_slider.setValue(0)
        self.head_pitch_slider.setValue(0)
        self._head_target_yaw=0
        self._head_target_pitch=0
        if self.robot and not self.head_tracking_enabled:
            async def go():
                try:
                    if hasattr(self.robot,"head_yaw"): await self.robot.head_yaw(0)
                    if hasattr(self.robot,"head_pitch"): await self.robot.head_pitch(0)
                except Exception: pass
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(go()))

    # ---------- Lights & Sound ----------
    def pick_color(self):
        c=QColorDialog.getColor(QColor(self.light_color), self, "Pick Color")
        if c.isValid():
            self.light_color=c.name()
            self.btn_color.setStyleSheet(f"background:{self.light_color}; color:#000;")
    def set_light_color(self):
        if not self.robot: return
        self.send_cmd(lambda: self.robot.neck_color(self.light_color))
        self.send_cmd(lambda: self.robot.left_ear_color(self.light_color))
        self.send_cmd(lambda: self.robot.right_ear_color(self.light_color))

    def say(self,word):
        if word == "hi":
            if self.use_personal_name: self.voice_say(f"hi {self.user_name}")
            else: self.voice_say("hi friend")
            return
        if self.robot and hasattr(self.robot,"say") and word in ("tada",):
            self.send_cmd(lambda: self.robot.say(word))
            return
        self.voice_say(word)

    def _speak_tts(self, text):
        try:
            self._tts_playing.set()
            if self.tts_engine is None:
                self.tts_engine = pyttsx3.init()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception:
            pass
        finally:
            self._tts_playing.clear()

    def _get_wav_duration(self, wav_path: str) -> float:
        try:
            with contextlib.closing(wave.open(wav_path,'rb')) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 1
                return max(0.0, frames / float(rate))
        except Exception:
            return 0.0

    def _mark_speaking_for_duration(self, seconds: float):
        if seconds <= 0:
            return
        self._tts_playing.set()
        def clear():
            try:
                time.sleep(seconds + 0.2)
            finally:
                self._tts_playing.clear()
        threading.Thread(target=clear, daemon=True).start()

    def _sys_play_wav(self, wav_path: str):
        # Windows
        if sys.platform.startswith("win"):
            try:
                import winsound
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass
            try:
                subprocess.Popen(["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        # macOS
        if sys.platform == "darwin":
            for cmd in (["afplay", wav_path], ["open", wav_path]):
                try:
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except Exception:
                    continue
        # Linux/Unix
        cmds = []
        if shutil.which("paplay"): cmds.append(["paplay", wav_path])
        if shutil.which("aplay"): cmds.append(["aplay", "-q", wav_path])
        if shutil.which("play"): cmds.append(["play", "-q", wav_path])
        if shutil.which("ffplay"): cmds.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", wav_path])
        if shutil.which("cvlc"): cmds.append(["cvlc", "--play-and-exit", "--quiet", wav_path])
        for cmd in cmds:
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
        try:
            import simpleaudio as sa
            wave_obj = sa.WaveObject.from_wave_file(wav_path)
            wave_obj.play()
        except Exception:
            pass

    def _ui_play_wav(self, wav_path: str):
        def _do():
            url = QUrl.fromLocalFile(os.path.abspath(wav_path))
            self.audio_player.setMedia(QMediaContent(url))
            self.audio_player.play()
            def maybe_fallback():
                try:
                    if self.audio_player.state() != QMediaPlayer.PlayingState:
                        self._sys_play_wav(wav_path)
                except Exception:
                    self._sys_play_wav(wav_path)
            QTimer.singleShot(600, maybe_fallback)
        QTimer.singleShot(0, _do)

    def voice_say(self, text):
        if getattr(self, "use_piper_tts", False) and self.piper and self.piper.available() and self.audio_on_pc_checkbox.isChecked():
            wav = self.piper.synth_to_wav(text)
            if wav and os.path.exists(wav) and os.path.getsize(wav) > 44:
                dur = self._get_wav_duration(wav)
                self._mark_speaking_for_duration(dur)
                self._ui_play_wav(wav)
                self.status_label.setText(f"SPEAK (Piper): {text}")
                return
            else:
                self.status_label.setText(f"Piper produced no audio. {getattr(self.piper, 'last_error', '')} Falling back to pyttsx3...")
                if self.audio_on_pc_checkbox.isChecked() and TTS_AVAILABLE:
                    threading.Thread(target=self._speak_tts, args=(text,), daemon=True).start()
                    return
                return
        if self.audio_on_pc_checkbox.isChecked() and TTS_AVAILABLE and not getattr(self, "disable_pyttsx3_fallback", True):
            threading.Thread(target=self._speak_tts, args=(text,), daemon=True).start()
            self.status_label.setText(f"SPEAK (PC): {text}")
            return
        self.status_label.setText(f"SPEAK: {text}")

    async def wave(self):
        if not self.robot: return
        try:
            await self.robot.head_yaw(25); await asyncio.sleep(0.25)
            await self.robot.head_yaw(-25); await asyncio.sleep(0.25)
            await self.robot.head_yaw(0)
        except Exception: pass

    async def spin_preset(self):
        if not self.robot: return
        try: await self.robot.spin(700); await asyncio.sleep(1.0); await self.robot.stop()
        except Exception: pass

    async def led_rainbow(self):
        if not self.robot: return
        colors=["#FF0000","#FF7F00","#FFFF00","#00FF00","#0000FF","#4B0082","#9400D3"]
        try:
            for c in colors:
                await self.robot.neck_color(c)
                if hasattr(self.robot,"left_ear_color"): await self.robot.left_ear_color(c)
                if hasattr(self.robot,"right_ear_color"): await self.robot.right_ear_color(c)
                await asyncio.sleep(0.15)
            await self.robot.neck_color("#000000")
            if hasattr(self.robot,"left_ear_color"): await self.robot.left_ear_color("#000000")
            if hasattr(self.robot,"right_ear_color"): await self.robot.right_ear_color("#000000")
        except Exception: pass

    async def led_blink(self):
        if not self.robot: return
        try:
            for _ in range(5):
                for c in ["#FFFFFF","#000000"]:
                    await self.robot.neck_color(c)
                    if hasattr(self.robot,"left_ear_color"): await self.robot.left_ear_color(c)
                    if hasattr(self.robot,"right_ear_color"): await self.robot.right_ear_color(c)
                    await asyncio.sleep(0.1)
        except Exception: pass

    # ---------- Sensors ----------
    def update_sensors(self):
        if self.robot and hasattr(self.robot,"sense"):
            try:
                moving=self.robot.is_moving()
                prox_l=self.robot.get_prox_left() if hasattr(self.robot,"get_prox_left") else 0
                prox_r=self.robot.get_prox_right() if hasattr(self.robot,"get_prox_right") else 0
                info=f"Moving:{moving} Prox:{prox_l}/{prox_r}"
                self.gui_bridge.sensor_update.emit(info)
                self.prox_left_bar.setValue(int(prox_l) if isinstance(prox_l,int) else 0)
                self.prox_right_bar.setValue(int(prox_r) if isinstance(prox_r,int) else 0)
            except Exception as e:
                self.gui_bridge.sensor_update.emit(f"Sensor err: {e}")
        else:
            self.gui_bridge.sensor_update.emit("Robot: --")

    # ---------- Face ----------
    def on_face_recog_toggle(self,state):
        self.face_manager.enabled=(state==Qt.Checked)
        self.face_status_label.setText(f"Face: {'ON' if self.face_manager.enabled else '--'}")

    def on_register_face(self):
        if not self.camera_active or self._last_rgb_frame is None:
            self.status_label.setText("Start camera first."); return
        frame = self._last_rgb_frame.copy()
        try: boxes = self.face_manager.detect_faces(frame)
        except Exception: boxes=[]
        box = max(boxes, key=lambda b: (b[2]-b[0])*(b[1]-b[3])) if boxes else self.stab.last_box
        def do_encode():
            try:
                pil = Image.fromarray(frame)
                rgb = np.ascontiguousarray(np.array(pil, dtype=np.uint8), dtype=np.uint8)
                locs = [box] if box else []
                if not locs and FACE_LIB_AVAILABLE:
                    locs = face_recognition.face_locations(rgb, model="hog")
                    if not locs: return (False, "No face found.")
                if FACE_LIB_AVAILABLE:
                    encs = face_recognition.face_encodings(rgb, locs)
                    if not encs: return (False, "Could not encode face.")
                    np.save(str(FACE_EMBED_FILE), encs[0]); return (True, "")
                else:
                    return (False, "face_recognition not installed.")
            except Exception as e:
                return (False, str(e))
        def after_encode(ok, msg):
            if ok:
                try: self.face_manager.user_encoding = np.load(str(FACE_EMBED_FILE))
                except Exception: pass
                self.status_label.setText("Face registered successfully.")
                self.voice_say(f"Okay {self.user_name}, I've registered your face.")
                self.face_status_label.setText("Face: Registered")
            else:
                self.status_label.setText(f"Reg error: {msg}")
        def worker():
            ok,msg = do_encode()
            QApplication.postEvent(self, _LambdaEvent(lambda: after_encode(ok,msg)))
        threading.Thread(target=worker, daemon=True).start()

    # ---------- Tablet ----------
    def on_tablet_detect_toggle(self,state):
        self.tablet_detect_enabled=(state==Qt.Checked)
        self.tablet_status_label.setText(f"Tablet: {'ON' if self.tablet_detect_enabled else '--'}")

    def maybe_tablet_voice(self, tid=None):
        now=time.time()
        if now - self.last_tablet_voice_time < TABLET_VOICE_COOLDOWN_SEC: return
        self.last_tablet_voice_time=now
        due=self.med_manager.due_now()
        id_text = f" tablet {tid}" if tid is not None else " tablet"
        if due:
            name,_,_=due[0]; self.voice_say(f"I see{id_text}. Is it time for your {name}?")
        else:
            self.voice_say(f"I see{id_text}. According to my schedule, no medicine is due now.")

    # ---------- Medication ----------
    def on_reminders_toggle(self,state):
        self.med_manager.enabled=(state==Qt.Checked)
        self.check_reminders()
        self.med_status_label.setText(f"Med: {'ON' if self.med_manager.enabled else 'OFF'}")
    def on_add_medicine(self):
        name,ok=QInputDialog.getText(self,"Medicine","Name:")
        if not ok or not name.strip(): return
        times_str,ok2=QInputDialog.getText(self,"Times","HH:MM (24h) or '5 pm', comma separated:")
        if not ok2: return
        times=[t.strip() for t in times_str.split(",") if t.strip()]
        success,cleaned=self.med_manager.add_medicine(name.strip(),times)
        if success:
            self.status_label.setText(f"Added {name}: {', '.join(cleaned)}")
            who = self.user_name if self.use_personal_name else "friend"
            self.voice_say(f"Okay {who}, I've added {name} to the schedule.")
            self.check_reminders()
        else:
            self.status_label.setText("No valid times provided.")
            self.voice_say("Sorry, please say times like 08:00 or 5 pm.")
    def on_show_schedule(self):
        if not self.med_manager.schedule:
            QMessageBox.information(self,"Schedule","No medicines scheduled."); return
        lines=[f"{k}: {', '.join(v)}" for k,v in self.med_manager.schedule.items()]
        QMessageBox.information(self,"Schedule","\n".join(lines))
    def check_reminders(self):
        self.med_manager.compute_next_upcoming()
        due=self.med_manager.due_now()
        if due:
            name,t,_=due[0]; self.med_status_label.setText(f"Med: Take {name} ({t})")
        else:
            self.med_status_label.setText(f"Med: {self.med_manager.next_upcoming or '--'}")

    # ---------- Recording ----------
    def toggle_video_recording(self, checked):
        if checked:
            self.start_video_recording()
        else:
            self.stop_video_recording()

    def start_video_recording(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = QFileDialog.getSaveFileName(self, "Save Video", f"dash_{now}.mp4", "Video (*.mp4 *.avi)")[0]
        if not fn:
            self.btn_record.setChecked(False)
            return
        w = int(self._cam_pending.get("width", 640))
        h = int(self._cam_pending.get("height", 480))
        fourcc = cv2.VideoWriter_fourcc(*('mp4v' if fn.lower().endswith('.mp4') else 'XVID'))
        self.video_writer = cv2.VideoWriter(fn, fourcc, 15, (w, h))
        if not self.video_writer.isOpened():
            self.status_label.setText("VideoWriter failed.")
            self.video_writer = None
            self.btn_record.setChecked(False)
            return
        self.recording = True
        self.recorded_video_path = fn
        self.btn_record.setText("Stop Recording")
        self.status_label.setText(f"Recording -> {fn}")

    def stop_video_recording(self):
        if getattr(self, "recording", False) and getattr(self, "video_writer", None):
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.status_label.setText(f"Saved: {self.recorded_video_path}")
        self.video_writer = None
        self.recording = False
        self.recorded_video_path = None
        self.btn_record.setText("Record")
        self.btn_record.setChecked(False)

    # ---------- ASR / Speech ----------
    def on_asr_backend_change(self, text):
        self.asr_backend = text

    def on_vosk_path_change(self):
        self.vosk_model_path = self.vosk_path_edit.text().strip()

    def on_speech_toggle(self, state):
        if state == Qt.Checked:
            self.start_speech()
        else:
            self.stop_speech()

    def start_speech(self):
        if self.speech_enabled:
            return
        backend = self.asr_backend
        if backend == "Vosk":
            if not VOSK_AVAILABLE:
                self.status_label.setText("Vosk not installed.")
                self.speech_checkbox.setChecked(False)
                return
            if not (self.vosk_model_path and os.path.isdir(self.vosk_model_path)):
                self.status_label.setText("Set valid VOSK model path in Settings.")
                self.speech_checkbox.setChecked(False)
                return
            self.speech_stop_event.clear()
            self.speech_enabled = True
            self.speech_thread = threading.Thread(target=self._vosk_worker, daemon=True)
            self.speech_thread.start()
            self.status_label.setText("Offline speech (Vosk) started.")
        else:
            if not SPEECH_AVAILABLE:
                self.status_label.setText("SpeechRecognition missing.")
                self.speech_checkbox.setChecked(False)
                return
            self.speech_stop_event.clear()
            self.speech_enabled = True
            self.speech_thread = threading.Thread(target=self._google_sr_worker, daemon=True)
            self.speech_thread.start()
            self.status_label.setText("Online speech (Google) started.")

    def stop_speech(self):
        if not self.speech_enabled:
            return
        self.speech_stop_event.set()
        if self.speech_thread:
            try:
                self.speech_thread.join(timeout=2)
            except Exception:
                pass
        self.speech_enabled = False
        self.status_label.setText("Speech stopped.")

    def _google_sr_worker(self):
        if not SPEECH_AVAILABLE:
            self.gui_bridge.status_update.emit("SpeechRecognition not available.")
            return
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.6
        r.non_speaking_duration = 0.35
        try:
            with sr.Microphone() as source:
                try:
                    r.adjust_for_ambient_noise(source, duration=1.0)
                except Exception:
                    pass
                while not self.speech_stop_event.is_set():
                    try:
                        audio = r.listen(source, timeout=2, phrase_time_limit=6)
                        if self.speech_stop_event.is_set():
                            break
                        if self._tts_playing.is_set():
                            continue
                        try:
                            text = r.recognize_google(audio, language=(self.google_language or "en-US"))
                        except sr.UnknownValueError:
                            continue
                        except sr.RequestError:
                            self.gui_bridge.status_update.emit("Google SR: network/problem.")
                            continue
                        if text:
                            self.route_nlu(text.lower().strip())
                    except sr.WaitTimeoutError:
                        continue
                    except Exception:
                        continue
        except Exception as e:
            self.gui_bridge.status_update.emit(f"Speech error: {e}")

    # VAD helper for Vosk
    def _vad_energy(self, samples, thresh=0.012):
        samples = np.abs(samples.astype(np.float32) / 32768.0)
        return np.sqrt(np.mean(samples**2)) > thresh

    def _vosk_worker(self):
        if not VOSK_AVAILABLE:
            self.gui_bridge.status_update.emit("Vosk not available.")
            return
        try:
            model = vosk.Model(self.vosk_model_path)
            import sounddevice as sd_local
            try:
                device_info = sd_local.query_devices(None, 'input')
                samplerate = int(device_info['default_samplerate'])
            except Exception:
                samplerate = 16000
            rec = vosk.KaldiRecognizer(model, samplerate)
            rec.SetWords(True)
            blocksize = int(samplerate * 0.5)

            def audio_callback(indata, frames, time_info, status):
                if self._tts_playing.is_set():
                    return
                if self.speech_stop_event.is_set():
                    raise sd_local.CallbackStop()
                ch0 = indata[:, 0]
                if abs(self.mic_gain - 1.0) > 1e-3:
                    ch0 = np.clip(ch0.astype(np.float32) * self.mic_gain, -32768, 32767).astype(np.int16)
                if not self._vad_energy(ch0, thresh=0.012):
                    return
                if rec.AcceptWaveform(ch0.tobytes()):
                    res = rec.Result()
                    try:
                        j = json.loads(res)
                        txt = (j.get("text") or "").strip()
                        if txt:
                            self.gui_bridge.status_update.emit(f"Heard: {txt}")
                            self.route_nlu(txt.lower())
                    except Exception:
                        pass
                else:
                    try:
                        p = json.loads(rec.PartialResult()).get("partial", "").strip()
                        if p:
                            self.gui_bridge.status_update.emit(f"Partial: {p}")
                    except Exception:
                        pass

            with sd_local.InputStream(
                channels=1,
                samplerate=samplerate,
                dtype='int16',
                callback=audio_callback,
                blocksize=blocksize,
            ):
                while not self.speech_stop_event.is_set():
                    sd_local.sleep(120)
        except Exception as e:
            self.gui_bridge.status_update.emit(f"Vosk error: {e}")

    # ---------- Ollama status ----------
    def update_ollama_status(self, manual=False):
        ver = _http_get_json(f"{self.ollama_url}/api/version")
        if not ver:
            self.ollama_status_label.setText("Ollama: unavailable")
            self.ollama_status_label.setStyleSheet("color:#ff6b6b;")
            if manual:
                self.status_label.setText("Ollama server not reachable.")
            return
        tags = _http_get_json(f"{self.ollama_url}/api/tags") or {}
        models = [m.get("name","") for m in tags.get("models",[])] if isinstance(tags,dict) else []
        if self.ollama_model in models or not self.ollama_model:
            self.ollama_status_label.setText(f"Ollama: connected ({ver.get('version','')}) • Model OK")
            self.ollama_status_label.setStyleSheet("color:#7bd88f;")
        else:
            self.ollama_status_label.setText(f"Ollama: connected ({ver.get('version','')}) • Model not pulled")
            self.ollama_status_label.setStyleSheet("color:#ffd166;")
            if manual:
                self.status_label.setText(f"Pull model: ollama pull {self.ollama_model}")

    def on_force_ollama_toggle(self, state):
        self.force_ollama_only = (state == Qt.Checked)
        mode = "Ollama only" if self.force_ollama_only else "Rule-based only"
        self.status_label.setText(f"NLU mode: {mode}")

    # --- High-priority rule: catch name changes even before NLU ---
    def _maybe_handle_name_change(self, text: str) -> bool:
        t = (text or "").strip().lower()
        patterns = [
            r'\bmy\s+name\s+is\s+(.+)$',
            r'\bcall\s+me\s+(.+)$',
            r"\bi'?m\s+(.+)$",
            r'\bi\s+am\s+(.+)$',
            r'\byou\s+can\s+call\s+me\s+(.+)$',
            r'\bthis\s+is\s+(.+)$',
        ]
        name = None
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                name = m.group(1)
                break
        if not name:
            return False
        name = re.split(r'[,.!?;]', name)[0].strip()
        name = re.sub(r'\b(please|ok|okay|thanks)\b$', '', name).strip()
        if not (1 <= len(name) <= 40):
            return False
        pretty = " ".join(w.capitalize() for w in re.split(r'\s+', name) if w)
        self.user_name = pretty
        self.user_profile["name"] = pretty
        self._save_user_profile(self.user_profile)
        self.voice_say(f"Okay, I will call you {self.user_name} from now on.")
        self.status_label.setText(f"Profile: name set to {self.user_name}")
        return True

    # ---------- Music / Choreo ----------
    def _load_choreo(self):
        if CHOREO_FILE.exists():
            try:
                with open(CHOREO_FILE,"r") as f: return json.load(f)
            except Exception: pass
        return [
            {"type":"sway","yaw":30,"duration":0.3},
            {"type":"sway","yaw":-30,"duration":0.3},
            {"type":"color","hex":"#FF3366"},
            {"type":"blink"},
            {"type":"spin","speed":500,"duration":0.35},
            {"type":"color","hex":"#33CCFF"},
        ]

    def _save_choreo(self):
        try:
            with open(CHOREO_FILE,"w") as f: json.dump(self.choreo_steps,f,indent=2)
        except Exception:
            pass

    def on_music_mode_toggle(self, state):
        en = (state == Qt.Checked)
        self.music_mode = en
        if en: self._start_beat_thread()
        else: self._stop_beat_thread()
        self.voice_say("Music mode on." if en else "Music mode off.")

    def _start_beat_thread(self):
        if getattr(self, "beat_thread", None) and self.beat_thread.is_alive():
            return
        self.beat_stop_event.clear()
        self.beat_thread = threading.Thread(target=self._beat_worker, daemon=True)
        self.beat_thread.start()

    def _stop_beat_thread(self):
        self.beat_stop_event.set()

    def _beat_worker(self):
        if not AUBIO_AVAILABLE:
            while not self.beat_stop_event.is_set():
                time.sleep(0.8)
                self._choreo_next_step()
            return
        samplerate=44100
        win_s=1024
        hop_s=512
        o = aubio.tempo("default", win_s, hop_s, samplerate)
        try:
            with sd.InputStream(channels=1, samplerate=samplerate, blocksize=hop_s, dtype='float32') as stream:
                while not self.beat_stop_event.is_set():
                    data, _ = stream.read(hop_s)
                    is_beat = o(np.array(data).astype(np.float32))
                    if is_beat:
                        self._choreo_next_step()
        except Exception as e:
            log.warning("Beat worker error: %s", e)
            while not self.beat_stop_event.is_set():
                time.sleep(0.9)
                self._choreo_next_step()

    def _choreo_next_step(self):
        steps = self.custom_choreo if self.custom_choreo else self.choreo_steps
        if not steps:
            return
        self._choreo_idx = (self._choreo_idx + 1) % len(steps)
        step = steps[self._choreo_idx]
        self._perform_step(step)

    def _perform_step(self, step):
        t = (step.get("type") or "").lower()
        if t == "sway":
            yaw = int(step.get("yaw", 30)); dur = float(step.get("duration", 0.25))
            async def go():
                try:
                    if self.robot and hasattr(self.robot,"head_yaw"):
                        await self.robot.head_yaw(yaw); await asyncio.sleep(dur)
                        await self.robot.head_yaw(-yaw); await asyncio.sleep(dur)
                        await self.robot.head_yaw(0)
                except Exception: pass
            self.send_cmd(go)
        elif t == "spin":
            sp = int(step.get("speed", 500)); dur=float(step.get("duration", 0.3))
            async def go():
                try:
                    if self.robot and hasattr(self.robot,"spin"):
                        await self.robot.spin(sp); await asyncio.sleep(dur); await self.robot.stop()
                except Exception: pass
            self.send_cmd(go)
        elif t == "blink":
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.led_blink()))
        elif t == "color":
            hexv = step.get("hex","#FFAA00")
            self.light_color = hexv
            self.set_light_color()
        elif t == "head":
            yaw = step.get("yaw"); pitch = step.get("pitch")
            async def go():
                try:
                    if self.robot:
                        if yaw is not None and hasattr(self.robot,"head_yaw"): await self.robot.head_yaw(int(yaw))
                        if pitch is not None and hasattr(self.robot,"head_pitch"): await self.robot.head_pitch(int(pitch))
                except Exception: pass
            self.send_cmd(go)

    # ---------- NLU routing ----------
    def route_nlu(self, text: str):
        if self._maybe_handle_name_change(text):
            return

        if self.force_ollama_only:
            obj = nlu_ollama_parse(text, self.ollama_url, self.ollama_model, timeout=10.0)
            actions = obj.get("actions") if isinstance(obj, dict) else None
            said_reply=False
            if actions and any(isinstance(a, dict) and a.get("type") != "noop" for a in actions):
                self.status_label.setText(f"NLU: Ollama • Heard: {text}")
                for act in actions:
                    if not isinstance(act, dict): continue
                    t = act.get("type","noop")

                    if t == "reply":
                        reply_text = (act.get("text") or "").strip()
                        if reply_text:
                            self.voice_say(reply_text)
                            said_reply=True
                        continue

                    if t == "set_name":
                        nm = (act.get("name") or "").strip()
                        if nm: self._maybe_handle_name_change(f"call me {nm}")
                        continue

                    if t == "notify":
                        msg=(act.get("text") or "").strip()
                        if msg: self.voice_say(msg)
                        continue

                    if t == "music_mode":
                        en=bool(act.get("enabled",True))
                        self.music_mode_checkbox.setChecked(en)
                        continue

                    if t == "choreo_start":
                        self.music_mode_checkbox.setChecked(True)
                        continue
                    if t == "choreo_stop":
                        self.music_mode_checkbox.setChecked(False)
                        continue
                    if t == "choreo_add_step":
                        step = (act.get("step") or "").strip().lower()
                        self._voice_add_step(step)
                        continue
                    if t == "choreo_clear":
                        self.custom_choreo.clear(); self._choreo_idx=0; self.voice_say("Cleared custom steps.")
                        continue

                    if t == "tablet_query":
                        self._voice_tablet_query()
                        continue

                    self._execute_action_dict(act)
                if not said_reply and (not actions or all(a.get("type") in ("noop","reply") for a in actions if isinstance(a,dict))):
                    reply = ollama_chat(text, self.ollama_url, self.ollama_model, timeout=12.0)
                    if reply: self.voice_say(reply)
                return
            else:
                reply = ollama_chat(text, self.ollama_url, self.ollama_model, timeout=12.0)
                if reply:
                    self.voice_say(reply)
                else:
                    self.voice_say("Sorry, I didn't understand that.")
                return

        self.status_label.setText(f"NLU: rules • Heard: {text}")
        return self.process_speech_intent(text)

    def _execute_action_dict(self, act: dict):
        t=act.get("type")
        if t=="move":
            direction=act.get("direction","forward"); sp=int(act.get("speed",250))
            if self.robot:
                if direction=="forward": self.send_cmd(lambda: self.robot.drive(sp))
                else: self.send_cmd(lambda: self.robot.drive(-sp))
            self.voice_say(f"Moving {direction}.")
        elif t=="turn":
            direction=act.get("direction","left"); sp=int(act.get("speed",350))
            val = sp if direction=="right" else -sp
            if self.robot: self.send_cmd(lambda: self.robot.spin(val))
            self.voice_say(f"Turning {direction}.")
        elif t=="stop":
            if self.robot: self.send_cmd(lambda: self.robot.stop()); self.voice_say("Stopping.")
        elif t=="head":
            yaw=act.get("yaw"); pitch=act.get("pitch")
            if yaw is not None and self.robot: self.send_cmd(lambda: self.robot.head_yaw(int(yaw)))
            if pitch is not None and self.robot: self.send_cmd(lambda: self.robot.head_pitch(int(pitch)))
        elif t=="head_track":
            en=bool(act.get("enabled",True)); self.head_track_checkbox.setChecked(en)
            self.voice_say("Head tracking on." if en else "Head tracking off.")
        elif t=="lights":
            mode=act.get("mode","set")
            if mode=="set":
                col=act.get("color","#FFAA00"); self.light_color=col; self.set_light_color(); self.voice_say("Color set.")
            elif mode=="party":
                en=bool(act.get("enabled",True))
                if en: self.light_color=hue_to_hex(random.random()); self.set_light_color(); self.voice_say("Party lights on.")
                else: self.light_color="#000000"; self.set_light_color(); self.voice_say("Party lights off.")
        elif t=="dance":
            en=bool(act.get("enabled",True))
            self.voice_say("Dance on." if en else "Dance off.")
        elif t=="choreo":
            name=act.get("name","Classic"); self.voice_say(f"{name} mode.")
        elif t=="teach_steps":
            self.voice_say("Okay! Let's learn some steps.")
        elif t=="face_register":
            self.on_register_face()
        elif t=="camera":
            a=act.get("action")
            if a=="on": self.btn_camera.setChecked(True); self.voice_say("Camera on.")
            elif a=="off": self.btn_camera.setChecked(False); self.voice_say("Camera off.")
            elif a=="capture": self.capture_image(); self.voice_say("Say cheese!")
            elif a=="record_on": self.btn_record.setChecked(True); self.voice_say("Recording.")
            elif a=="record_off": self.btn_record.setChecked(False); self.voice_say("Stopped recording.")
        elif t=="tablet_detect":
            en=bool(act.get("enabled",True)); self.tablet_detect_checkbox.setChecked(en)
            self.voice_say("Tablet detection on." if en else "Tablet detection off.")
        elif t=="tablet_mode":
            mode=act.get("mode","ArUco")
            self.tablet_mode_combo.setCurrentText("ArUco" if mode.lower().startswith("aru") and ARUCO_AVAILABLE else "Color")
            self.voice_say(f"Tablet mode {mode}.")
        elif t=="med_add":
            name=act.get("name","").strip(); times=act.get("times",[])
            if name and times:
                ok,_=self.med_manager.add_medicine(name,times)
                if ok: self.check_reminders(); self.voice_say(f"Added {name}.")
                else: self.voice_say("Please say times like 08:00 or 5 pm.")
        elif t=="med_reminders":
            en=bool(act.get("enabled",True)); self.reminders_checkbox.setChecked(en); self.voice_say("Reminders on." if en else "Reminders off.")
        elif t=="med_mark_taken":
            ok,info=self.med_manager.ack_first_due()
            if ok: self.check_reminders(); name,t=info; self.voice_say(f"Marked {name} at {t} as taken.")
            else: self.voice_say("No medicine due right now.")
        elif t=="med_query_due":
            due=self.med_manager.due_now()
            if due: name,_,_=due[0]; self.voice_say(f"It is time for your {name}.")
            else: self.voice_say("No medicine is due right now.")
        elif t=="med_show_schedule":
            if not self.med_manager.schedule: self.voice_say("No medicines scheduled.")
            else:
                parts=[f"{k}: {', '.join(v)}" for k,v in self.med_manager.schedule.items()]
                self.voice_say("Your schedule is: " + " ; ".join(parts))

    # ---------- Rule-based fallback intents ----------
    def process_speech_intent(self,text):
        if self._maybe_handle_name_change(text):
            return

        if text.startswith("add medicine"):
            m=re.match(r'add medicine\s+(.+?)\s+at\s+(.+)$', text)
            if m:
                name=m.group(1).strip()
                raw=m.group(2)
                times=re.findall(r'(\d{1,2}:\d{2}\s*(?:am|pm)?)', raw) or re.findall(r'(\d{1,2}\s*(?:am|pm))', raw)
                if not times:
                    times=[t.strip() for t in raw.split(",") if t.strip()]
                success,_=self.med_manager.add_medicine(name,times)
                if success: self.voice_say(f"Added {name}."); self.check_reminders()
                else: self.voice_say("Please say times like 08:00 or 5 pm.")
                return
        if "reminders on" in text or "enable reminders" in text:
            self.reminders_checkbox.setChecked(True); self.voice_say("Reminders are on."); return
        if "reminders off" in text or "disable reminders" in text:
            self.reminders_checkbox.setChecked(False); self.voice_say("Reminders are off."); return
        if "mark taken" in text or "i took my medicine" in text:
            ok,info=self.med_manager.ack_first_due()
            if ok: name,t=info; self.voice_say(f"Marked {name} at {t} as taken."); self.check_reminders()
            else: self.voice_say("No medicine due right now to mark as taken.")
            return
        if "show schedule" in text or "what is my schedule" in text or "what's my schedule" in text:
            if not self.med_manager.schedule: self.voice_say("No medicines scheduled.")
            else:
                parts=[f"{k}: {', '.join(v)}" for k,v in self.med_manager.schedule.items()]
                self.voice_say("Your schedule is: " + " ; ".join(parts))
            return
        if "what are my reminders" in text or "my reminders" in text or "next reminder" in text:
            due=self.med_manager.due_now()
            if due:
                names=", ".join([n for n,_,_ in due]); self.voice_say(f"Due now: {names}.")
            elif self.med_manager.next_upcoming:
                self.voice_say(f"Your next reminder is {self.med_manager.next_upcoming}.")
            else:
                self.voice_say("You have no upcoming reminders.")
            return

        if "move forward" in text or "go forward" in text or re.search(r'\bforward\b', text):
            if self.robot: self.send_cmd(lambda: self.robot.drive(250)); self.voice_say("Moving forward."); return
        if "move backward" in text or "go back" in text or re.search(r'\bback(ward)?\b', text):
            if self.robot: self.send_cmd(lambda: self.robot.drive(-220)); self.voice_say("Going backward."); return
        if "turn left" in text or re.search(r'\bleft\b', text):
            if self.robot: self.send_cmd(lambda: self.robot.spin(-350)); self.voice_say("Turning left."); return
        if "turn right" in text or re.search(r'\bright\b', text):
            if self.robot: self.send_cmd(lambda: self.robot.spin(350)); self.voice_say("Turning right."); return
        if re.search(r'\bstop\b', text) and "dance" not in text:
            if self.robot: self.send_cmd(lambda: self.robot.stop()); self.voice_say("Stopping."); return

        if "look left" in text and self.robot:
            self.send_cmd(lambda: self.robot.head_yaw(-70)); return
        if "look right" in text and self.robot:
            self.send_cmd(lambda: self.robot.head_yaw(70)); return
        if "look up" in text and self.robot:
            self.send_cmd(lambda: self.robot.head_pitch(-10)); return
        if "look down" in text and self.robot:
            self.send_cmd(lambda: self.robot.head_pitch(15)); return
        if "center head" in text or "reset head" in text:
            self.center_head(); self.voice_say("Centering my head."); return

        if "track my face" in text or "follow me" in text or "head track on" in text:
            self.head_track_checkbox.setChecked(True); self.voice_say("Okay, I'll track your face."); return
        if "stop tracking" in text or "stop following" in text or "head track off" in text:
            self.head_track_checkbox.setChecked(False); self.voice_say("Head tracking is off."); return

        COLOR_KEYWORDS = {
            "red":"#FF0000","green":"#00FF00","blue":"#0000FF","pink":"#FF69B4",
            "purple":"#8000FF","yellow":"#FFFF00","orange":"#FF7F00","cyan":"#00FFFF",
            "white":"#FFFFFF","black":"#000000"
        }
        if "party lights" in text and "disable" not in text:
            self.light_color=hue_to_hex(random.random()); self.set_light_color(); self.voice_say("Party time!"); return
        if "lights off" in text or "disable party lights" in text:
            self.light_color="#000000"; self.set_light_color(); self.voice_say("Okay, lights off."); return
        if "set light color to" in text or "change color to" in text:
            for name,hexv in COLOR_KEYWORDS.items():
                if re.search(r'\b'+re.escape(name)+r'\b', text):
                    self.light_color=hexv; self.set_light_color(); self.voice_say(f"Color set to {name}."); return

        if "tablet detection on" in text:
            self.tablet_detect_checkbox.setChecked(True); self.voice_say("Tablet detection on."); return
        if "tablet detection off" in text:
            self.tablet_detect_checkbox.setChecked(False); self.voice_say("Tablet detection off."); return
        if "tablet mode aruco" in text and ARUCO_AVAILABLE:
            self.tablet_mode_combo.setCurrentText(TABLET_MODE_ARUCO); self.voice_say("Aruco mode selected."); return
        if "tablet mode color" in text:
            self.tablet_mode_combo.setCurrentText(TABLET_MODE_COLOR); self.voice_say("Color mode selected."); return
        if "which tablet" in text or "tablet active" in text:
            self._voice_tablet_query(); return

        if "register my face" in text or "scan my face" in text:
            self.on_register_face(); return
        if "start camera" in text:
            self.btn_camera.setChecked(True); self.voice_say("Camera is on."); return
        if "stop camera" in text:
            self.btn_camera.setChecked(False); self.voice_say("Camera off."); return
        if "take picture" in text or "take photo" in text:
            self.capture_image(); self.voice_say("Say cheese!"); return
        if "start recording" in text:
            self.btn_record.setChecked(True); self.voice_say("Recording started."); return
        if "stop recording" in text:
            self.btn_record.setChecked(False); self.voice_say("Recording stopped."); return

        if "dance to music" in text or "music mode on" in text:
            self.music_mode_checkbox.setChecked(True); return
        if "music mode off" in text or "stop dancing" in text:
            self.music_mode_checkbox.setChecked(False); return
        if re.search(r'\badd (a )?dance step\b', text):
            m = re.search(r'add (?:a )?dance step\s+(.+)$', text)
            if m:
                self._voice_add_step(m.group(1).strip().lower())
            return
        if "clear dance steps" in text or "clear choreo" in text:
            self.custom_choreo.clear(); self._choreo_idx=0; self.voice_say("Cleared custom steps."); return

        if self.use_llm_nlu:
            reply = ollama_chat(text, self.ollama_url, self.ollama_model, timeout=12.0)
            if reply:
                self.voice_say(reply)
                return

        self.status_label.setText(f"Unrecognized speech: {text}")

    def _voice_add_step(self, spec: str):
        step=None
        if spec in ("spin left","spin_left"):
            step={"type":"spin","speed":-500,"duration":0.35}
        elif spec in ("spin right","spin_right"):
            step={"type":"spin","speed":500,"duration":0.35}
        elif spec.startswith("color"):
            m=re.search(r'color\s*:\s*(#[0-9a-f]{6})', spec)
            if not m:
                named={"red":"#FF0000","green":"#00FF00","blue":"#0000FF","pink":"#FF69B4","purple":"#8000FF","yellow":"#FFFF00","orange":"#FF7F00","cyan":"#00FFFF","white":"#FFFFFF"}
                for k,v in named.items():
                    if re.search(r'\b'+k+r'\b', spec):
                        step={"type":"color","hex":v}
                        break
            else:
                step={"type":"color","hex":m.group(1)}
        elif spec.startswith("head"):
            m=re.search(r'head\s*:\s*yaw\s*:\s*(-?\d+)', spec) or re.search(r'head\s*yaw\s*(-?\d+)', spec)
            if m: step={"type":"head","yaw":int(m.group(1))}
            m2=re.search(r'head\s*:\s*pitch\s*:\s*(-?\d+)', spec) or re.search(r'head\s*pitch\s*(-?\d+)', spec)
            if m2: step={"type":"head","pitch":int(m2.group(1))}
        elif "sway" in spec:
            m=re.search(r'sway\s*(-?\d+)?', spec)
            yaw=int(m.group(1)) if m and m.group(1) else 30
            step={"type":"sway","yaw":yaw,"duration":0.3}
        elif "blink" in spec:
            step={"type":"blink"}
        if step:
            self.custom_choreo.append(step); self._choreo_idx=0
            self._save_choreo()
            self.voice_say("Added a dance step.")
        else:
            self.voice_say("Sorry, I couldn't parse that step.")

    def _voice_tablet_query(self):
        if self.last_tablet_id is not None:
            self.voice_say(f"Tablet {self.last_tablet_id} is active.")
        else:
            mode = self.tablet_mode
            if self.tablet_detect_enabled:
                self.voice_say(f"No ArUco ID in view. Tablet mode is {mode}.")
            else:
                self.voice_say("Tablet detection is off.")

    # ---------- Close ----------
    def closeEvent(self,event):
        try: self.stop_camera()
        except Exception: pass
        try: self.stop_speech()
        except Exception: pass
        try: self._stop_beat_thread()
        except Exception: pass
        super().closeEvent(event)

# Helper to post back to UI thread
class _LambdaEvent(QEvent):
    _etype = QEvent.Type(QEvent.registerEventType())
    def __init__(self, cb): super().__init__(self._etype); self.cb=cb

def customEvent(self, event):
    if isinstance(event, _LambdaEvent):
        try: event.cb()
        except Exception: pass

# ---------- Entrypoint ----------
def main():
    QWidget.customEvent = customEvent
    app=QApplication(sys.argv)
    w=DashFullController()
    w.show()
    return app.exec_()

if __name__=="__main__":
    try:
        print("Launching Dash Robot Controller...", flush=True)
        sys.exit(main())
    except Exception as e:
        import traceback
        print("Fatal error starting UI:", e, flush=True)
        traceback.print_exc()
        sys.exit(1)
