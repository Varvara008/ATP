import eel, base64, os
from predict import load_model, predict

eel.init("client")
model = load_model("defect_model.pth")

@eel.expose
def process_file(file, file_name):

    print("file", file, "name", file_name)
    with open(f"media/{file_name}", "wb") as f:
        f.write(base64.b64decode(file))
    
    result = predict(model, f"media/{file_name}")
    return result
    


if __name__ == "__main__":
    eel.start("index.html", size=(600, 600))

