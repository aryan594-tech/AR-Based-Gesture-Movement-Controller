import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ============================================================
# Load VFX PNGs (must be in same folder: repulsor.png, shockwave.png, particle.png)
# ============================================================
def load_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

REPULSOR_PNG = load_png("repulsor.png")
SHOCKWAVE_PNG = load_png("shockwave.png")
PARTICLE_PNG = load_png("particle.png")


def overlay_png(frame, png, center, scale=1.0, alpha_mul=1.0):
    """
    Overlay an RGBA png on BGR frame at a given center (x, y),
    scaled by 'scale', multiplied alpha by alpha_mul.
    """
    if png is None:
        return

    h, w, _ = frame.shape
    ph, pw = png.shape[:2]

    new_w = int(pw * scale)
    new_h = int(ph * scale)
    if new_w <= 0 or new_h <= 0:
        return

    resized = cv2.resize(png, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if resized.shape[2] == 3:
        # No alpha: treat as solid
        overlay = resized
        mask = np.ones((new_h, new_w), dtype=np.uint8) * int(255 * alpha_mul)
    else:
        overlay = resized[:, :, :3]
        mask = resized[:, :, 3]
        # multiply alpha
        mask = (mask.astype(np.float32) * alpha_mul).astype(np.uint8)

    x, y = center
    x1 = int(x - new_w / 2)
    y1 = int(y - new_h / 2)
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Clip to frame
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return

    x1_clip = max(x1, 0)
    y1_clip = max(y1, 0)
    x2_clip = min(x2, w)
    y2_clip = min(y2, h)

    overlay_x1 = x1_clip - x1
    overlay_y1 = y1_clip - y1
    overlay_x2 = overlay_x1 + (x2_clip - x1_clip)
    overlay_y2 = overlay_y1 + (y2_clip - y1_clip)

    roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]
    overlay_roi = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    mask_roi = mask[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    mask_f = mask_roi.astype(np.float32) / 255.0
    inv_mask = 1.0 - mask_f

    roi_f = roi.astype(np.float32)
    overlay_f = overlay_roi.astype(np.float32)

    blended = (overlay_f * mask_f[..., None] + roi_f * inv_mask[..., None])
    frame[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)


# ============================================================
# MediaPipe setup
# ============================================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def lm_point(landmarks, idx, w, h):
    try:
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)
    except Exception:
        return None


def draw_glow_line(img, p1, p2, color, thickness=4, alpha=0.6):
    if p1 is None or p2 is None:
        return
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_glow_circle(img, center, radius, color, alpha=0.7):
    if center is None:
        return
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_glow_poly(img, pts, color, alpha=0.4):
    pts = [p for p in pts if p is not None]
    if len(pts) < 3:
        return
    pts_np = np.array(pts, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_np], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ============================================================
# Suit drawing (same base as before)
# ============================================================
def draw_body_suit(frame, pose_lm, mode="NORMAL"):
    h, w, _ = frame.shape

    if mode == "NORMAL":
        main_color = (0, 255, 255)
        accent_color = (255, 0, 255)
        chest_color = (0, 200, 255)
    else:
        main_color = (0, 0, 255)
        accent_color = (0, 140, 255)
        chest_color = (0, 50, 255)

    sh_l = lm_point(pose_lm, 11, w, h)
    sh_r = lm_point(pose_lm, 12, w, h)
    hip_l = lm_point(pose_lm, 23, w, h)
    hip_r = lm_point(pose_lm, 24, w, h)
    elbow_l = lm_point(pose_lm, 13, w, h)
    elbow_r = lm_point(pose_lm, 14, w, h)
    wrist_l = lm_point(pose_lm, 15, w, h)
    wrist_r = lm_point(pose_lm, 16, w, h)
    knee_l = lm_point(pose_lm, 25, w, h)
    knee_r = lm_point(pose_lm, 26, w, h)
    ankle_l = lm_point(pose_lm, 27, w, h)
    ankle_r = lm_point(pose_lm, 28, w, h)

    if sh_l and sh_r and hip_l and hip_r:
        torso_poly = [sh_l, sh_r, hip_r, hip_l]
        draw_glow_poly(frame, torso_poly, chest_color, alpha=0.35)

        chest_top_mid = ((sh_l[0] + sh_r[0]) // 2, (sh_l[1] + sh_r[1]) // 2)
        chest_bottom_mid = ((hip_l[0] + hip_r[0]) // 2, (hip_l[1] + hip_r[1]) // 2)
        cx = chest_top_mid[0]
        cy = (chest_top_mid[1] + chest_bottom_mid[1]) // 2
        radius = max(18, int(abs(chest_top_mid[1] - chest_bottom_mid[1]) * 0.18))
        draw_glow_circle(frame, (cx, cy), radius, (255, 255, 255), alpha=0.9)
        draw_glow_circle(frame, (cx, cy), int(radius * 0.5), main_color, alpha=0.9)

    draw_glow_line(frame, sh_l, elbow_l, main_color, thickness=7 if mode == "COMBAT" else 5)
    draw_glow_line(frame, elbow_l, wrist_l, main_color, thickness=7 if mode == "COMBAT" else 5)
    draw_glow_line(frame, sh_r, elbow_r, main_color, thickness=7 if mode == "COMBAT" else 5)
    draw_glow_line(frame, elbow_r, wrist_r, main_color, thickness=7 if mode == "COMBAT" else 5)
    draw_glow_circle(frame, wrist_l, 18, accent_color, alpha=0.7)
    draw_glow_circle(frame, wrist_r, 18, accent_color, alpha=0.7)

    draw_glow_line(frame, hip_l, knee_l, main_color, thickness=8 if mode == "COMBAT" else 6)
    draw_glow_line(frame, knee_l, ankle_l, main_color, thickness=8 if mode == "COMBAT" else 6)
    draw_glow_line(frame, hip_r, knee_r, main_color, thickness=8 if mode == "COMBAT" else 6)
    draw_glow_line(frame, knee_r, ankle_r, main_color, thickness=8 if mode == "COMBAT" else 6)

    draw_glow_line(frame, hip_l, hip_r, accent_color, thickness=6, alpha=0.7)
    draw_glow_line(frame, sh_l, sh_r, accent_color, thickness=6, alpha=0.7)


def draw_face_helmet(frame, face_lm, mode="NORMAL"):
    h, w, _ = frame.shape
    if mode == "NORMAL":
        helmet_color = (0, 255, 255)
    else:
        helmet_color = (0, 0, 255)

    points = []
    idxs = [10, 151, 9, 0, 152, 234, 454]
    for i in idxs:
        try:
            lm = face_lm.landmark[i]
            points.append((int(lm.x * w), int(lm.y * h)))
        except Exception:
            pass
    if len(points) >= 3:
        draw_glow_poly(frame, points, helmet_color, alpha=0.25)

    for i in [33, 133, 362, 263]:
        try:
            lm = face_lm.landmark[i]
            x, y = int(lm.x * w), int(lm.y * h)
            draw_glow_circle(frame, (x, y), 6, (255, 255, 255), alpha=0.9)
        except Exception:
            pass


def draw_hand_armor(frame, hand_lm, mode="NORMAL"):
    h, w, _ = frame.shape
    color = (255, 0, 255) if mode == "NORMAL" else (0, 127, 255)
    for lm in hand_lm.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        draw_glow_circle(frame, (x, y), 6, color, alpha=0.8)


def wrists_above_shoulders(pose_lm, w, h):
    try:
        sh_l = lm_point(pose_lm, 11, w, h)
        sh_r = lm_point(pose_lm, 12, w, h)
        wrist_l = lm_point(pose_lm, 15, w, h)
        wrist_r = lm_point(pose_lm, 16, w, h)
        if None in (sh_l, sh_r, wrist_l, wrist_r):
            return False
        shoulder_y = min(sh_l[1], sh_r[1])
        return wrist_l[1] < shoulder_y and wrist_r[1] < shoulder_y
    except Exception:
        return False


# ============================================================
# VFX state: particles + shockwaves
# ============================================================
class Particle:
    def __init__(self, x, y):
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(1, 4)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed
        self.x = x
        self.y = y
        self.life = np.random.uniform(0.6, 1.2)
        self.age = 0.0

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.age += dt

    @property
    def alive(self):
        return self.age < self.life

    @property
    def alpha(self):
        return max(0.0, 1.0 - self.age / self.life)


class Shockwave:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.age = 0.0
        self.life = 0.7

    def update(self, dt):
        self.age += dt

    @property
    def alive(self):
        return self.age < self.life

    @property
    def scale(self):
        # grows from 0.4 to 1.8
        t = self.age / self.life
        return 0.4 + 1.4 * t

    @property
    def alpha(self):
        t = self.age / self.life
        return max(0.0, 1.0 - t)


# ============================================================
# Main loop
# ============================================================
def main():
    cap = cv2.VideoCapture(0)
    mode = "NORMAL"
    last_toggle_time = 0

    # For punch detection
    prev_right_wrist = None
    prev_time = time.time()

    particles = []
    shockwaves = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        now = time.time()
        dt = now - prev_time
        prev_time = now

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(img_rgb)

        pose_lm = res.pose_landmarks.landmark if res.pose_landmarks else None

        # --- Suit + mode toggle ---
        if pose_lm:
            if wrists_above_shoulders(pose_lm, w, h):
                if now - last_toggle_time > 1.0:
                    mode = "COMBAT" if mode == "NORMAL" else "NORMAL"
                    last_toggle_time = now

            draw_body_suit(frame, pose_lm, mode=mode)

        # --- Face helmet ---
        if res.face_landmarks:
            draw_face_helmet(frame, res.face_landmarks, mode=mode)

        # --- Hand armor & VFX anchors ---
        right_palm_center = None

        if res.right_hand_landmarks:
            draw_hand_armor(frame, res.right_hand_landmarks, mode=mode)
            # Palm-ish point: landmark 9
            rh = res.right_hand_landmarks.landmark
            cx = int(rh[9].x * w)
            cy = int(rh[9].y * h)
            right_palm_center = (cx, cy)

            # Spawn cyberpunk particles around right hand
            for _ in range(2):
                particles.append(Particle(cx, cy))

        if res.left_hand_landmarks:
            draw_hand_armor(frame, res.left_hand_landmarks, mode=mode)
            lh = res.left_hand_landmarks.landmark
            lx = int(lh[9].x * w)
            ly = int(lh[9].y * h)
            for _ in range(2):
                particles.append(Particle(lx, ly))

        # --- Iron Man repulsor (always visible on right palm if PNG exists) ---
        if right_palm_center and REPULSOR_PNG is not None:
            overlay_png(frame, REPULSOR_PNG, right_palm_center,
                        scale=0.7, alpha_mul=0.9)

        # --- Punch detection: fast right wrist movement -> shockwave ---
        if pose_lm:
            rw = lm_point(pose_lm, 16, w, h)
            if prev_right_wrist is not None and rw is not None and dt > 0:
                vx = (rw[0] - prev_right_wrist[0]) / dt
                vy = (rw[1] - prev_right_wrist[1]) / dt
                speed = math.hypot(vx, vy)

                # Threshold for "punch"
                if speed > 900:  # tweak if too sensitive
                    # Spawn shockwave near wrist
                    shockwaves.append(Shockwave(rw[0], rw[1]))
            prev_right_wrist = rw

        # --- Update and render particles ---
        alive_particles = []
        for p in particles:
            p.update(dt)
            if p.alive:
                alive_particles.append(p)
                alpha = p.alpha
                if PARTICLE_PNG is not None:
                    overlay_png(frame, PARTICLE_PNG,
                                (int(p.x), int(p.y)),
                                scale=0.3,
                                alpha_mul=alpha)
                else:
                    draw_glow_circle(frame,
                                     (int(p.x), int(p.y)),
                                     4,
                                     (255, 0, 255),
                                     alpha=alpha)
        particles = alive_particles

        # --- Update and render shockwaves ---
        alive_shocks = []
        for s in shockwaves:
            s.update(dt)
            if s.alive:
                alive_shocks.append(s)
                if SHOCKWAVE_PNG is not None:
                    overlay_png(frame, SHOCKWAVE_PNG,
                                (int(s.x), int(s.y)),
                                scale=s.scale,
                                alpha_mul=s.alpha)
                else:
                    draw_glow_circle(frame,
                                     (int(s.x), int(s.y)),
                                     int(40 * s.scale),
                                     (0, 255, 255),
                                     alpha=s.alpha)
        shockwaves = alive_shocks

        # --- HUD ---
        hud_text = f"MODE: {mode} | Repulsor + Particles + Punch Shockwave"
        color = (0, 255, 255) if mode == "NORMAL" else (0, 0, 255)
        cv2.putText(frame, hud_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "Raise both hands above shoulders to toggle mode | Punch forward for shockwave",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

        cv2.imshow("AR Suit + VFX", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Display the yellow_circles.png image from assets folder
img = cv2.imread('assets/yellow_circles.png')
if img is not None:
    cv2.imshow('Yellow Circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Could not load assets/yellow_circles.png')
