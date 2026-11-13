import time
log_file = "reports/stage2run3_dryrun.log"
while True:
    with open(log_file, "r") as f:
        text = f.read()
    if "Training finished!" in text or "Traceback" in text or "Error" in text:
        break
    time.sleep(2)
