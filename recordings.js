let currentlyPlayingAudio = null;
let currentlyPlayingButton = null;

function togglePlay(button) {
    const card = button.closest(".recording-card");
    const audio = card.querySelector("audio");

    if (currentlyPlayingAudio && currentlyPlayingAudio !== audio) {
        currentlyPlayingAudio.pause();
        if (currentlyPlayingButton) {
            currentlyPlayingButton.textContent = "Play";
        }
    }

    if (audio.paused) {
        audio.play();
        button.textContent = "Pause";
        currentlyPlayingAudio = audio;
        currentlyPlayingButton = button;
    } else {
        audio.pause();
        button.textContent = "Play";
        currentlyPlayingAudio = null;
        currentlyPlayingButton = null;
    }

    audio.onended = function () {
        button.textContent = "Play";
        currentlyPlayingAudio = null;
        currentlyPlayingButton = null;
    };
}
