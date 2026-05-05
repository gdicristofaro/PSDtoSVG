import os
import subprocess
import shutil

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths based on script directory (which is flask/)
project_root = os.path.dirname(script_dir)  # PSDtoSVG/
src_psdtosvg = os.path.join(project_root, 'src', 'psdtosvg')
frontend_dir = os.path.join(project_root, 'frontend')
flask_static = os.path.join(script_dir, 'static')
frontend_dist = os.path.join(frontend_dir, 'dist')
venv_dir = os.path.join(script_dir, '.venv')
requirements_file = os.path.join(script_dir, 'requirements.txt')
python_executable = shutil.which('python') or shutil.which('python3')

# 1. Run python -m build in src/psdtosvg
print("Building psdtosvg package...")
subprocess.run([python_executable, '-m', 'build'], cwd=src_psdtosvg, check=True)

# 2. In frontend directory: npm install and npm run build-server-frontend
print("Installing npm dependencies...")
subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)

print("Building frontend...")
subprocess.run(['npm', 'run', 'build-server-frontend'], cwd=frontend_dir, check=True)

# 3. Replace contents of flask/static with contents of frontend/dist
print("Replacing static files...")
if os.path.exists(flask_static):
    shutil.rmtree(flask_static)
shutil.copytree(frontend_dist, flask_static)

# 4. Establish venv if not already established
if not os.path.exists(venv_dir):
    print("Creating virtual environment...")
    subprocess.run([python_executable, '-m', 'venv', venv_dir], check=True)

# Activate venv (by using its python) and reinstall packages
print("Installing Python requirements...")
venv_python = os.path.join(venv_dir, 'bin', 'python')
subprocess.run([venv_python, '-m', 'pip', 'install', '--force-reinstall', '-r', requirements_file], check=True)

print("Build process completed successfully!")