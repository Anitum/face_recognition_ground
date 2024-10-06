import tkinter as tk
import subprocess
import threading
import queue

class FaceRecognitionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition Control Panel")

        self.start_button = tk.Button(master, text="Start Face Recognition", command=self.start_recognition)
        self.start_button.pack()

        self.stop_button = tk.Button(master, text="Stop Face Recognition", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack()

        self.status_label = tk.Label(master, text="Status: Idle")
        self.status_label.pack()

        self.output_text = tk.Text(master, height=10, width=50)
        self.output_text.pack()

        self.process = None
        self.output_queue = queue.Queue()

    def start_recognition(self):
        if not self.process:
            self.process = subprocess.Popen(['python', 'archi/production_code_final.py'], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.STDOUT,
                                            universal_newlines=True)
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running")

            threading.Thread(target=self.read_output, daemon=True).start()
            self.update_output()

    def stop_recognition(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")

    def read_output(self):
        for line in self.process.stdout:
            self.output_queue.put(line)

    def update_output(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                self.output_text.insert(tk.END, line)
                self.output_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.update_output)

root = tk.Tk()
gui = FaceRecognitionGUI(root)
root.mainloop()