import eel, base64, os

eel.init("client")

@eel.expose
def process_file(file, file_name):

    print("file", file, "name", file_name)
    with open(f"media/{file_name}", "wb") as f:
        f.write(base64.b64decode(file))

if __name__ == "__main__":
    eel.start("index.html", size=(600, 600))