# ===================== IMPORTS =====================
import cv2
import pygame
import numpy as np
import pandas as pd
import mediapipe as mp
from threading import Thread
from matplotlib import pyplot as plt
import sqlite3
plt.ion()
# ===================== CONFIG =====================
WEBCAM = 0
MEDIA_ABERTURA_PADRAO = 0.45
JANELA_MEDIA = 10
QNT_FRAMES_CONSECUTIVOS_ALARME_ON = 40
TEMPO_ALARME = 2200

ALARME_ON = False
CONTADOR_QUADROS_SONOLENCIA = 0

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
# ================= DATABASE =================
conn = sqlite3.connect("fatigue_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS fatigue_records (
id INTEGER PRIMARY KEY AUTOINCREMENT,
eye_value REAL,
fatigue_level TEXT
)
""")

conn.commit()

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

            cv2.putText(frame, f"Media: {media_abertura:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            # ================= FATIGUE LEVEL =================
            if media_abertura > 0.60:
                level = "Normal"
                recommendation = "Status: Normal - Stay focused"

            elif 0.45 < media_abertura <= 0.60:
                level = "Slight Fatigue"
                recommendation = "Recommendation: Take a short break"

            else:
                level = "High Fatigue"
                recommendation = "Recommendation: Drink water and rest"

            fatigue_percent = max(0, min(100, int((0.7 - media_abertura) * 150)))

            cv2.putText(frame, f"Fatigue: {fatigue_percent}%",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(255,0,255),2)
            cv2.putText(frame, recommendation,
                        (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,255),2)            
                        

            # Save data every 10 frames
            if len(dados) % 10 == 0:
                cursor.execute(
                    "INSERT INTO fatigue_records (eye_value, fatigue_level) VALUES (?, ?)",
                    (media_abertura, level)
                )
                conn.commit()
            if media_abertura < MEDIA_ABERTURA_PADRAO:
                CONTADOR_QUADROS_SONOLENCIA += 1

                if CONTADOR_QUADROS_SONOLENCIA >= QNT_FRAMES_CONSECUTIVOS_ALARME_ON:
                    if not ALARME_ON:
                        ALARME_ON = True
                        Thread(target=alerta_sonoro, daemon=True).start()

                    cv2.putText(frame, "ALERTA FADIGA!",
                                (200, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
            else:
                CONTADOR_QUADROS_SONOLENCIA = 0
                ALARME_ON = False

        cv2.imshow("Detector de Fadiga", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# ===================== SAVE & CLEAN =====================
dados.to_csv("dados_abertura_olhos.csv", index=False)
cap.release()
cv2.destroyAllWindows()
conn.close()

plt.plot(dados["Media_Abertura_Olhos"])
plt.title("Media de Abertura dos Olhos")
plt.xlabel("Frames")
plt.ylabel("Abertura")
plt.grid()
plt.show()
