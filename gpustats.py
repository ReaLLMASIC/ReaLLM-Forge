import time
import gpustat
import csv
import os

def monitor_gpu(interval=5):
    csv_file_path = 'gpustats_log.csv'

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Time', 'GPU ID', 'Name', 'Utilization (%)', 'Memory Used (MB)', 'Total Memory (MB)', 'Temperature (°C)'])
            file.flush()

        try:
            while True:
                stats = gpustat.new_query()
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                for gpu in stats.gpus:
                    writer.writerow([
                        current_time,
                        gpu.index,
                        gpu.name,
                        gpu.utilization,
                        gpu.memory_used,
                        gpu.memory_total,
                        gpu.temperature
                    ])
                    print(f"[{current_time}] GPU {gpu.index} - {gpu.name}: {gpu.utilization}% utilization, {gpu.memory_used} MB / {gpu.memory_total} MB memory used, {gpu.temperature} °C")
                file.flush()
                time.sleep(interval)

        except KeyboardInterrupt:
            print("Monitoring stopped.")

monitor_gpu(interval=2)
