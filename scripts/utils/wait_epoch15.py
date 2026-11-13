import time
log_file = "reports/stage2run3_train.log"
while True:
    with open(log_file, "r") as f:
        text = f.read()
    if "Epoch 15 Summary" in text or "Traceback" in text:
        break
    time.sleep(10)
