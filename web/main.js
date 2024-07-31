document.addEventListener("DOMContentLoaded", () => {
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('fileElem');

    dropArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropArea.classList.add('highlight');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('highlight');
    });

    dropArea.addEventListener('drop', (event) => {
        event.preventDefault();
        dropArea.classList.remove('highlight');
        const files = event.dataTransfer.files;
        handleFiles(files);
    });

    fileElem.addEventListener('change', () => {
        handleFiles(fileElem.files);
    });
});

async function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];

        // Display the image immediately
        const displayImage = document.getElementById('display-image');
        displayImage.src = URL.createObjectURL(file);
        displayImage.style.display = 'block';

        // Perform calculations in the background
        const data = await readFileAsBase64(file);
        const caption = await eel.get_caption(data)();
        const scores = await eel.score_image(data)();

        document.getElementById('caption').textContent = caption;
        document.getElementById('max-similarity').textContent = `Max Similarity: ${scores.max_similarity}%`;
        document.getElementById('avg-similarity').textContent = `Avg Similarity: ${scores.avg_similarity}%`;
        document.getElementById('median-similarity').textContent = `Median Similarity: ${scores.median_similarity}%`;
    }
}

function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
}
