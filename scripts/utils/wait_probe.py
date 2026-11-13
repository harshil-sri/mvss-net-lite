import time
import os

log_file = "/home/harshil/.gemini/antigravity-cli/brain/c5e2e37d-2ff8-4c4a-b29e-552f13f1c0d6/.system_generated/tasks/task-213.log"
while True:
    with open(log_file, "r") as f:
        text = f.read()
    if "Evaluating: model/checkpoints/stage2run2_mvss_lite_ep50.pt" in text and "MIDV500-Auth FP (0.9):" in text.split("Evaluating: model/checkpoints/stage2run2_mvss_lite_ep50.pt")[-1]:
        print(text)
        break
    time.sleep(2)
