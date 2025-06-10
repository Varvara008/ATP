import eel, base64, os, sys
from predict import load_model, predict
from bottle import static_file, route, run
from get_path import resource_path

eel.init("client")
model = load_model(resource_path("defect_model.pth"))

media_dir = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), "media")

@eel.expose
def process_file(file, file_name):
    file_path = os.path.join(media_dir, file_name)
    print("file", file, "name", file_name)

    with open(file_path, "wb") as f:
        f.write(base64.b64decode(file))
    
    result = predict(model, f"media/{file_name}")
    return result
    
@route("/media/<filename>")
def serve_media(filename):
    return static_file(filename, root="media")


if __name__ == "__main__":
    eel.start("index.html", size=(750, 1000))
    run(host="localhost", port=8000, quiet=True)
