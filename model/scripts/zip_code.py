import os
import zipfile

def create_zip():
    zip_name = 'mvss_net_lite_code.zip'
    exclude_dirs = [
        'data',
        'data_pipeline/raw',
        'data_pipeline/processed',
        'model/checkpoints',
        '.git',
        '__pycache__',
        'swarm_reports',
        'graphify-out'
    ]
    exclude_files = [
        'swarm.py',
        'swarm.log',
        zip_name
    ]
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if not any(root.replace('./', '') + '/' + d == ex or d == ex or (root.replace('./', '') + '/' + d).startswith(ex + '/') for ex in exclude_dirs)]
            
            for file in files:
                if file in exclude_files:
                    continue
                file_path = os.path.join(root, file)
                # Don't zip the tar.gz I made earlier
                if file_path.endswith('.tar.gz'):
                    continue
                zipf.write(file_path, arcname=file_path)
    print(f"Created {zip_name} successfully.")

if __name__ == '__main__':
    create_zip()
