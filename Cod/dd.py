# ===================== IMPORTS =====================
import cv2
import pygame
import numpy as np
import pandas as pd
import mediapipe as mp
import sqlite3
from threading import Thread
from matplotlib import pyplot as plt
from datetime import datetime

plt.ion()

# ===================== CONFIG =====================
WEBCAM = 0
MEDIA_ABERTURA_PADRAO = 0.45
JANELA_MEDIA = 10
QNT_FRAMES_CONSECUTIVOS_ALARME_ON = 40
TEMPO_ALARME = 2200

ALARME_ON = False
CONTADOR_QUADROS_SONOLENCIA = 0

# ===================== DATABASE =====================
conn = sqlite3.connect("fatigue_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS fatigue_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_abertura REAL,
    fatigue_level TEXT,
    timestamp TEXT
)
""")
conn.commit()

# ===================== MEDIAPIPE =====================
mp_face_mesh = mp.solutions.face_mesh

# ===================== EYE LANDMARKS =====================
OLHO_DIREITO = [362, 382, 381, 380, 374, 373, 390, 249,
                263, 466, 388, 387, 386, 385, 384, 398]

OLHO_ESQUERDO = [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]

# ===================== DATA =====================
dist_dir, dist_esq = [], []
dados = pd.DataFrame(columns=["Media_Abertura_Olhos"])

# ===================== AUDIO =====================
pygame.init()
pygame.mixer.init()
ALARME = pygame.mixer.Sound("alarme.wav")

def alerta_sonoro():
    global ALARME_ON
    while ALARME_ON:
        ALARME.play()
        pygame.time.wait(TEMPO_ALARME)

# ===================== FUNCTIONS =====================

def calcular_altura_olhos(pontos):
    A = np.linalg.norm(pontos[1] - pontos[15])
    B = np.linalg.norm(pontos[2] - pontos[14])
    C = np.linalg.norm(pontos[3] - pontos[13])
    D = np.linalg.norm(pontos[4] - pontos[12])
    E = np.linalg.norm(pontos[5] - pontos[11])
    F = np.linalg.norm(pontos[6] - pontos[10])
    G = np.linalg.norm(pontos[7] - pontos[9])
    H = np.linalg.norm(pontos[0] - pontos[8])
    return (A+B+C+D+E+F+G) / (2.0*H)

# -------- FATIGUE CLASSIFICATION --------
def classify_fatigue(media):
    if media > 0.55:
        return "LOW"
    elif 0.45 <= media <= 0.55:
        return "MEDIUM"
    else:
        return "HIGH"

# -------- SUGGESTION MODULE --------
def give_suggestion(level):
    if level == "LOW":
        return "Low Fatigue: Take 10 mins break and drink water."
    elif level == "MEDIUM":
        return "Medium Fatigue: Take short nap and eat energy food."
    else:
        return "High Fatigue: Sleep 1.5 hours and do stretching."

# ===================== CAMERA =====================
cap = cv2.VideoCapture(WEBCAM)

# ===================== MAIN LOOP =====================
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            pontos = np.array([
                [int(p.x * w), int(p.y * h)]
                for p in results.multi_face_landmarks[0].landmark
            ])

            olho_dir = pontos[OLHO_DIREITO]
            olho_esq = pontos[OLHO_ESQUERDO]

            ear_dir = calcular_altura_olhos(olho_dir)
            ear_esq = calcular_altura_olhos(olho_esq)

            dist_dir.append(ear_dir)
            dist_esq.append(ear_esq)

            if len(dist_dir) > JANELA_MEDIA:
                dist_dir.pop(0)
                dist_esq.pop(0)

            media_abertura = (np.mean(dist_dir) + np.mean(dist_esq)) / 2
            dados.loc[len(dados)] = media_abertura

            # CLASSIFICATION
            fatigue_level = classify_fatigue(media_abertura)
            suggestion = give_suggestion(fatigue_level)

            # SAVE TO DATABASE
            cursor.execute(
                "INSERT INTO fatigue_records (media_abertura, fatigue_level, timestamp) VALUES (?, ?, ?)",
                (media_abertura, fatigue_level, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()

            # DISPLAY
            cv2.putText(frame, f"Media: {media_abertura:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Level: {fatigue_level}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            cv2.putText(frame, suggestion,
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            # ALARM LOGIC
            if fatigue_level == "HIGH":
                CONTADOR_QUADROS_SONOLENCIA += 1

                if CONTADOR_QUADROS_SONOLENCIA >= QNT_FRAMES_CONSECUTIVOS_ALARME_ON:
                    if not ALARME_ON:
                        ALARME_ON = True
                        Thread(target=alerta_sonoro, daemon=True).start()
            else:
                CONTADOR_QUADROS_SONOLENCIA = 0
                ALARME_ON = False

        cv2.imshow("AI Mental Fatigue Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# ===================== SAVE & CLEAN =====================
dados.to_csv("dados_abertura_olhos.csv", index=False)
conn.close()
cap.release()
cv2.destroyAllWindows()

plt.plot(dados["Media_Abertura_Olhos"])
plt.title("Eye Opening Average Over Time")
plt.xlabel("Frames")
plt.ylabel("Opening Value")
plt.grid()
plt.show()