const dropZone = document.querySelector(".drop-zone");

["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("highlight");
    }, false)
});

["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("highlight");
    }, false)
});

dropZone.addEventListener("drop", e => {
    let file = e.dataTransfer.files[0];
    if(file){
        if(!file.type.startsWith("image/")){
            dropZone.querySelector("p").textContent = "Загружать можно только изображения!";
            dropZone.classList.add("error");
            setTimeout(() => {
                dropZone.querySelector("p").textContent = "Перетащите изображение сюда";
                dropZone.classList.remove("error");
            }, 1000);
            return;
        }
        dropZone.querySelector("p").textContent = `Загружен файл:${file.name}. Анализирую...`;
        let reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            let byteData = reader.result.split(",")[1];
            eel.process_file(byteData, file.name)();
        }

    }
})