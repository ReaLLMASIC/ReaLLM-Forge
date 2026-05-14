import cv2
import ollama
import base64
import threading
import time

# --- CONFIG ---
MODEL = "gemma4:e4b"
VIDEO_INDEX = 0      # AUKEY Camera
# Square resolution optimized for Vision Transformers (ViT)
RES_W, RES_H = 200, 200 

class VisionAssistant:
    def __init__(self):
        # Initialize camera with Jetson-optimized V4L2 backend
        self.cap = cv2.VideoCapture(VIDEO_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
        
        self.latest_frame = None
        self.is_running = True
        
        # Start the Producer thread (Capture)
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        """Constantly updates the internal buffer to prevent stale frames."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Force resize if the camera doesn't natively support 600x600
                self.latest_frame = cv2.resize(frame, (RES_W, RES_H))
            time.sleep(0.01) 

    def ask(self, prompt):
        """Grabs the 'Now' frame and queries the LLM."""
        if self.latest_frame is None:
            print("Camera not ready...")
            return

        # 1. Encode the current frame to JPEG at slightly lower quality to save bandwidth
        #
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', self.latest_frame, encode_param)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        print(f"\n[Analyzing {RES_W}x{RES_H} Frame...]")
        
        try:
            # 2. Stream the response
            stream = ollama.chat(
                model=MODEL,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }],
                stream=True,
            )

            print("Assistant: ", end="", flush=True)
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print("\n" + "-"*30)
            
        except Exception as e:
            print(f"Ollama Error: {e}")

    def start(self):
        print(f"--- Orin Live Vision Active ({RES_W}x{RES_H}) ---")
        print("Type your question and press Enter. (Type 'q' to quit)")
        
        while True:
            user_input = input("\nQuestion > ")
            if user_input.lower() == 'q':
                self.is_running = False
                break
            
            # Default prompt if you just hit Enter
            prompt = user_input if user_input.strip() != "" else "What is happening in front of you?"
            self.ask(prompt)

        self.cap.release()

if __name__ == "__main__":
    VisionAssistant().start()
