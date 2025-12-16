import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import os
import joblib
import numpy as np
from collections import deque
import warnings
import pyautogui
import time
import threading
from tensorflow.keras.models import load_model

# Tắt cảnh báo và cấu hình hệ thống
warnings.filterwarnings("ignore")
pyautogui.FAILSAFE = False 
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class GestureApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ===== CẤU HÌNH CỬA SỔ =====
        self.title("🎮 AI Gesture Control - High FPS Optimized")
        self.geometry("1300x750")
        
        # ===== LOAD MODEL =====
        try:
            self.model = load_model("data/hand_gesture_mlp.h5")
            self.scaler = joblib.load("data/scaler.pkl")
            self.le = joblib.load("data/label_encoder.pkl")
        except Exception as e:
            print(f"Lỗi load model: {e}")
            self.model = None

        # ===== CẤU HÌNH HÀNH ĐỘNG =====
        self.GESTURE_ACTIONS = {
            "thumb up": ("🔊 Tăng Âm", "volume_up"),
            "fist": ("🔉 Giảm Âm", "volume_down"),
            "peace": ("⬆️ Cuộn Lên", "scroll_up"),
            "ok": ("⬇️ Cuộn Xuống", "scroll_down"),
            "palm": ("⏯️ Play/Pause", "pause_play"),
            "index up": ("🖱️ Chuột Trái", "left_click"),
            "call": ("🖱️ Chuột Phải", "right_click"),
            "rock": ("⏭️ Bài Tiếp", "next_track"),
            "gun sign": ("⏮️ Bài Trước", "prev_track"),
            "L sign": ("📸 Chụp Ảnh", "screenshot"),
            "C sign": ("📋 Copy", "copy"),
            "cross fingers": ("🔄 Tab", "alt_tab"),
            "3 fingers up": ("🔆 Tăng Sáng", "brightness_up"),
        }
        self.gesture_enabled = {k: True for k in self.GESTURE_ACTIONS}

        # ===== BIẾN TRẠNG THÁI & TỐI ƯU =====
        self.is_running = False
        self.control_active = True
        
        # Threading
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None  # Frame đã xử lý để hiển thị
        self.latest_gesture_text = "None"
        self.latest_action_text = "---"
        
        # Frame Skipping (Tối ưu FPS)
        self.frame_count = 0
        self.SKIP_FRAMES = 3 # Chỉ chạy model AI mỗi 3 frame
        
        # Kéo thả & Chuột
        self.is_dragging = False
        self.pinch_start_thresh = 0.035
        self.pinch_stop_thresh = 0.05
        self.mouse_queue = deque(maxlen=5) # Smoothing
        self.frame_margin = 100
        
        # Nhận diện
        self.proba_buffer = deque(maxlen=5)
        self.last_action_time = {}
        self.cooldown = 2

        # ===== MEDIAPIPE =====
        self.mp_hands = mp.solutions.hands
        # Giảm độ tin cậy xuống một chút để bắt tay nhanh hơn
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.setup_ui()

    def setup_ui(self):
        # (Giữ nguyên giao diện cũ của bạn)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === SIDEBAR ===
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(self.sidebar, text="GESTURE CONTROLLER", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#00ff41").grid(row=0, column=0, padx=20, pady=20)

        self.btn_camera = ctk.CTkButton(self.sidebar, text="▶ BẬT CAMERA", command=self.toggle_camera,
                                        fg_color="#00ff41", text_color="black", font=("Arial", 14, "bold"))
        self.btn_camera.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.tabview = ctk.CTkTabview(self.sidebar)
        self.tabview.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_settings = self.tabview.add("Cài Đặt")
        self.tab_gestures = self.tabview.add("Cử Chỉ")

        self.sw_control = ctk.CTkSwitch(self.tab_settings, text="Kích hoạt Điều khiển chuột", 
                                        command=self.toggle_control_status)
        self.sw_control.select()
        self.sw_control.pack(pady=15, padx=10, anchor="w")

        ctk.CTkLabel(self.tab_settings, text="Độ mượt chuột:").pack(pady=(10,0), anchor="w")
        self.slider_smooth = ctk.CTkSlider(self.tab_settings, from_=1, to=15, number_of_steps=14,
                                           command=lambda v: setattr(self.mouse_queue, 'maxlen', int(v)))
        self.slider_smooth.set(5)
        self.slider_smooth.pack(pady=5, padx=10, fill="x")

        # Gestures List
        self.scroll_gestures = ctk.CTkScrollableFrame(self.tab_gestures, label_text="Bật/Tắt Cử Chỉ")
        self.scroll_gestures.pack(fill="both", expand=True)
        for gesture, (desc, _) in self.GESTURE_ACTIONS.items():
            row = ctk.CTkFrame(self.scroll_gestures, fg_color="transparent")
            row.pack(fill="x", pady=2)
            sw = ctk.CTkSwitch(row, text=f"{gesture}", font=("Arial", 11),
                               command=lambda g=gesture: self.toggle_gesture_state(g))
            sw.select()
            sw.pack(side="left")

        # === CAMERA ===
        self.main_frame = ctk.CTkFrame(self, fg_color="#000000")
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.lbl_camera = ctk.CTkLabel(self.main_frame, text="Sẵn sàng...", text_color="gray")
        self.lbl_camera.pack(fill="both", expand=True)

        self.status_bar = ctk.CTkFrame(self, height=40, fg_color="#1a1a2e")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.lbl_status_gesture = ctk.CTkLabel(self.status_bar, text="Cử chỉ: None", 
                                               font=("Arial", 14, "bold"), text_color="#00ff41")
        self.lbl_status_gesture.pack(side="left", padx=20)
        self.lbl_status_action = ctk.CTkLabel(self.status_bar, text="Hành động: ---", 
                                              font=("Arial", 14), text_color="white")
        self.lbl_status_action.pack(side="left", padx=20)

        # Thêm hiển thị FPS
        self.lbl_fps = ctk.CTkLabel(self.status_bar, text="FPS: 0", font=("Arial", 12), text_color="gray")
        self.lbl_fps.pack(side="right", padx=20)

    # ===== LOGIC UI =====
    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            # Giảm độ phân giải đầu vào để tăng tốc xử lý (nhưng vẫn đủ nét cho AI)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            self.btn_camera.configure(text="⏹ DỪNG CAMERA", fg_color="#ff4757")
            
            # Bắt đầu luồng xử lý video riêng biệt
            self.thread = threading.Thread(target=self.video_process_loop, daemon=True)
            self.thread.start()
            
            # Bắt đầu vòng lặp cập nhật giao diện
            self.update_gui_loop()
        else:
            self.is_running = False
            if self.cap: self.cap.release()
            self.btn_camera.configure(text="▶ BẬT CAMERA", fg_color="#00ff41")
            self.lbl_camera.configure(image=None, text="Camera đã tắt")

    def toggle_control_status(self): self.control_active = bool(self.sw_control.get())
    def toggle_gesture_state(self, gesture): self.gesture_enabled[gesture] = not self.gesture_enabled[gesture]

    # ===== LOGIC XỬ LÝ (CHẠY TRÊN THREAD RIÊNG) =====
    def get_smooth_mouse_coords(self, raw_x, raw_y, frame_w, frame_h):
        screen_w, screen_h = pyautogui.size()
        x_mapped = np.interp(raw_x, (self.frame_margin, frame_w - self.frame_margin), (0, screen_w))
        y_mapped = np.interp(raw_y, (self.frame_margin, frame_h - self.frame_margin), (0, screen_h))
        self.mouse_queue.append((x_mapped, y_mapped))
        avg_x = sum(p[0] for p in self.mouse_queue) / len(self.mouse_queue)
        avg_y = sum(p[1] for p in self.mouse_queue) / len(self.mouse_queue)
        return np.clip(avg_x, 0, screen_w), np.clip(avg_y, 0, screen_h)

    def perform_action(self, gesture_name):
        if not self.control_active or not self.gesture_enabled.get(gesture_name, False): return "Đã tắt"
        if time.time() - self.last_action_time.get(gesture_name, 0) < self.cooldown: return "Chờ..."
        
        desc, code = self.GESTURE_ACTIONS.get(gesture_name, ("", ""))
        try:
            if code == "volume_up": pyautogui.press('volumeup')
            elif code == "volume_down": pyautogui.press('volumedown')
            elif code == "scroll_up": pyautogui.scroll(300)
            elif code == "scroll_down": pyautogui.scroll(-300)
            elif code == "pause_play": pyautogui.press('playpause')
            elif code == "left_click": pyautogui.click()
            elif code == "right_click": pyautogui.rightClick()
            elif code == "next_track": pyautogui.press('nexttrack')
            elif code == "prev_track": pyautogui.press('prevtrack')
            elif code == "screenshot": pyautogui.hotkey('win', 'shift', 's')
            elif code == "copy": pyautogui.hotkey('ctrl', 'c')
            elif code == "alt_tab": pyautogui.hotkey('alt', 'tab')
            elif code == "brightness_up": pyautogui.press('brightnessup')
            self.last_action_time[gesture_name] = time.time()
            return desc
        except: return "Lỗi"

    def video_process_loop(self):
        """Vòng lặp xử lý nặng: Đọc Cam -> AI -> Điều khiển"""
        prev_time = 0
        current_gesture_display = "None"
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: continue

            # Lật và chuyển màu
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            # MediaPipe xử lý
            results = self.hands.process(rgb)
            
            gesture_text = "None"
            action_text = "---"

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # Tọa độ ngón
                thumb = hand_lms.landmark[4]
                index = hand_lms.landmark[8]
                ix, iy = int(index.x * w), int(index.y * h)
                tx, ty = int(thumb.x * w), int(thumb.y * h)

                # --- LOGIC CHUỘT (Chạy mỗi frame để mượt) ---
                if self.control_active:
                    # Kéo thả
                    dist = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
                    if not self.is_dragging and dist < self.pinch_start_thresh:
                        pyautogui.mouseDown()
                        self.is_dragging = True
                    elif self.is_dragging and dist > self.pinch_stop_thresh:
                        pyautogui.mouseUp()
                        self.is_dragging = False
                    
                    # Di chuyển chuột
                    sx, sy = self.get_smooth_mouse_coords(ix, iy, w, h)
                    pyautogui.moveTo(sx, sy)

                    # Vẽ UI
                    color = (0, 0, 255) if self.is_dragging else (0, 255, 255)
                    cv2.circle(frame, (ix, iy), 8, color, -1)
                    if self.is_dragging:
                        cv2.line(frame, (ix, iy), (tx, ty), color, 2)
                        cv2.putText(frame, "DRAGGING", (ix+10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- NHẬN DIỆN CỬ CHỈ (Chạy mỗi 3 frame) ---
                # Chỉ chạy khi không kéo thả để tiết kiệm CPU
                if not self.is_dragging and self.model:
                    self.frame_count += 1
                    if self.frame_count % self.SKIP_FRAMES == 0:
                        data = []
                        for lm in hand_lms.landmark: data.extend([lm.x, lm.y, lm.z])
                        
                        if len(data) == 63:
                            X = self.scaler.transform(np.array([data]))
                            prob = self.model.predict(X, verbose=0)[0]
                            self.proba_buffer.append(prob)
                            avg_prob = np.mean(self.proba_buffer, axis=0)
                            idx = np.argmax(avg_prob)
                            
                            if avg_prob[idx] > 0.6:
                                gesture_name = self.le.inverse_transform([idx])[0]
                                current_gesture_display = f"{gesture_name} ({int(avg_prob[idx]*100)}%)"
                                action_text = self.perform_action(gesture_name)
                            else:
                                current_gesture_display = "Unknown"
                    
                    gesture_text = current_gesture_display
            
            else:
                if self.is_dragging: # An toàn: nhả chuột nếu mất tay
                    pyautogui.mouseUp()
                    self.is_dragging = False

            # Tính FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # Cập nhật dữ liệu chia sẻ cho luồng UI
            with self.lock:
                self.latest_frame = frame
                self.latest_gesture_text = gesture_text
                self.latest_action_text = action_text
                self.fps_text = f"FPS: {int(fps)}"

            time.sleep(0.001) # Nhường CPU

    def update_gui_loop(self):
        """Vòng lặp cập nhật giao diện (nhẹ nhàng)"""
        if self.is_running:
            with self.lock:
                frame = self.latest_frame
                g_text = self.latest_gesture_text
                a_text = self.latest_action_text
                fps_val = getattr(self, 'fps_text', "0")

            if frame is not None:
                # Resize ảnh hiển thị (dùng Nearest để nhanh nhất)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Resize thông minh giữ tỷ lệ
                disp_w = self.main_frame.winfo_width()
                disp_h = self.main_frame.winfo_height()
                if disp_w > 10:
                    img.thumbnail((disp_w, disp_h), Image.Resampling.NEAREST)
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self.lbl_camera.configure(image=ctk_img, text="")
                
                # Cập nhật text
                self.lbl_status_gesture.configure(text=f"Cử chỉ: {g_text}")
                self.lbl_status_action.configure(text=f"Hành động: {a_text}")
                self.lbl_fps.configure(text=fps_val)

            self.after(30, self.update_gui_loop) # 33ms ~ 30 FPS cho UI là đủ

    def on_closing(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = GestureApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()