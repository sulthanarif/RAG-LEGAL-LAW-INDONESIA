from flask import Flask, render_template, request, jsonify, send_file
import os
import random
import shutil

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SORTED_DIR = os.path.join(BASE_DIR, "data", "pdf")      # Base path for sorted files (target categories)
TRASH_DIR = os.path.join(BASE_DIR, "data", "trash")

# Ensure directories exist
os.makedirs(SORTED_DIR, exist_ok=True)
os.makedirs(TRASH_DIR, exist_ok=True)

CATEGORIES = ["peraturan-menteri", "peraturan-pemerintah", "peraturan-presiden", "undang-undang"]
for cat in CATEGORIES:
    os.makedirs(os.path.join(SORTED_DIR, cat), exist_ok=True)

# Simpan status file yang sudah diproses agar tidak muncul lagi di sesi ini
processed_files = set()

def get_unsorted_files():
    files = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(SORTED_DIR, cat)
        if os.path.exists(cat_dir):
            for f in os.listdir(cat_dir):
                if f.endswith('.pdf') and f not in processed_files:
                    files.append({"filename": f, "folder": cat})
    return files

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/next_doc")
def next_doc():
    files = get_unsorted_files()
    if not files:
        return jsonify({"status": "empty", "message": "Proses sortir selesai."})
    
    selected = random.choice(files)
    return jsonify({
        "status": "success",
        "filename": selected["filename"],
        "folder": selected["folder"],
        "categories": CATEGORIES
    })

@app.route("/api/preview/<folder>/<filename>")
def preview(folder, filename):
    file_path = os.path.join(SORTED_DIR, folder, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='application/pdf')
    return "Not found", 404

@app.route("/api/action", methods=["POST"])
def action():
    data = request.json
    action_type = data.get("action")
    filename = data.get("filename")
    original_folder = data.get("original_folder")
    
    source_path = os.path.join(SORTED_DIR, original_folder, filename)
    
    if not os.path.exists(source_path) and action_type != "skip":
        return jsonify({"status": "error", "message": "File tidak ditemukan."})
        
    try:
        if action_type == "move":
            category = data.get("category")
            new_name = data.get("new_name", filename)
            if not new_name.endswith('.pdf'):
                new_name += '.pdf'
                
            dest_dir = os.path.join(SORTED_DIR, category)
            dest_path = os.path.join(dest_dir, new_name)
            shutil.move(source_path, dest_path)
            processed_files.add(filename)
            message = f"Berhasil dipindahkan ke {category}!"
            
        elif action_type == "rename_only":
            new_name = data.get("new_name", filename)
            if not new_name.endswith('.pdf'):
                new_name += '.pdf'
            dest_path = os.path.join(SORTED_DIR, original_folder, new_name)
            shutil.move(source_path, dest_path)
            processed_files.add(filename)
            message = "Nama berhasil diganti!"
            return jsonify({"status": "success", "message": message, "new_filename": new_name})

        elif action_type == "delete":
            dest_path = os.path.join(TRASH_DIR, filename)
            shutil.move(source_path, dest_path)
            processed_files.add(filename)
            message = "Dokumen dihapus (ke Trash)."
            
        elif action_type == "skip":
            processed_files.add(filename)
            message = "Dokumen dilewati."
            
        return jsonify({"status": "success", "message": message})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    print("🚀 Game Sortir Dokumen berjalan di http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
