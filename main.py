import eel, base64, os

eel.init("client")

@eel.expose
def process_file(file, file_name):

    print("file", file, "name", file_name)

if __name__ == "__main__":
    eel.start("index.html", size=(600, 600))