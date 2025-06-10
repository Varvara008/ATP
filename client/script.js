const dropZone = document.querySelector(".drop-zone");
const result_text = document.querySelector(".result-text");

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
            eel.process_file(byteData, file.name)().then(res=>{
                result_text.textContent = (res.label == 0) ? "присутствует" : "отсутствует";
                dropZone.querySelector("p").textContent = `Загружен файл:${file.name}. Ответ готов`;
            })
            
        }

    }
})