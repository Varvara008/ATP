import eel, base64, os
from predict import load_model, predict
from bottle import static_file, route, run

eel.init("client")
model = load_model("defect_model.pth")

@eel.expose
def process_file(file, file_name):

    print("file", file, "name", file_name)
    with open(f"media/{file_name}", "wb") as f:
        f.write(base64.b64decode(file))
    
    result = predict(model, f"media/{file_name}")
    return result
    
@route("/media/<filename>")
def serve_media(filename):
    return static_file(filename, root="media")

if __name__ == "__main__":
    eel.start("index.html", size=(750, 1000))
    run(host="localhost", port=8000, quiet=True)
