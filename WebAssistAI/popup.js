document.addEventListener("DOMContentLoaded", () => {
  async function callProcessPdfApi(url, query) {
      const apiUrl = "http://localhost:8000/process_pdf"; // Replace with your actual API endpoint
      document.getElementById("loadingSpinner").style.display = "block"; // Show spinner

      try {
          const response = await fetch(apiUrl, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ content_url: url, user_query: query }),
          });

          if (!response.ok) {
              throw new Error(`API Error: ${response.status} - ${response.statusText}`);
          }

          const data = await response.json();
          return data;
      } catch (error) {
          console.error("Error calling API:", error);
          return { status: "error", answer: error.message };
      } finally {
          document.getElementById("loadingSpinner").style.display = "none"; // Hide spinner
      }
  }

  document.getElementById("queryBtn").addEventListener("click", () => {
      const query = document.getElementById("queryInput").value.trim();
      const responseContainer = document.getElementById("responseContainer");

      if (!query) {
          responseContainer.innerText = "Error: Please enter a query.";
          return;
      }

      responseContainer.innerText = ""; // Clear previous response
      document.getElementById("loadingSpinner").style.display = "block"; // Show spinner

      chrome.runtime.sendMessage({ action: "getURL" }, async (response) => {
          const url = response.url;
          if (!url) {
              responseContainer.innerText = "Error: No active tab found.";
              document.getElementById("loadingSpinner").style.display = "none"; // Hide spinner
              return;
          }

          const apiResponse = await callProcessPdfApi(url, query);
          responseContainer.innerText = apiResponse.status === "success" ? apiResponse.answer : `Error: ${apiResponse.answer}`;
      });
  });

  let mediaRecorder;
let audioChunks = [];

document.getElementById("micBtn").addEventListener("click", async () => {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Setup MediaRecorder
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        audioChunks = [];

        // When audio data is available, push it into audioChunks
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        // Stop recording when silence is detected (3s silence)
        detectSilence(stream, () => {
            mediaRecorder.stop();
        });

        // When recording stops, save the file
        mediaRecorder.onstop = () => {
            saveAudioFile(audioChunks);
        };

        console.log("🎤 Recording started...");
    } catch (err) {
        console.error("Microphone access denied:", err);
        alert("Microphone access is required!");
    }
});

function saveAudioFile(audioChunks) {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const audioUrl = URL.createObjectURL(audioBlob);
    
    // Create a link to download the file
    const link = document.createElement("a");
    link.href = audioUrl;
    link.download = "recording.wav";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    console.log("🎙️ Audio saved as recording.wav");
}

// Detect silence in microphone input
function detectSilence(stream, callback, silenceDelay = 3000) {
    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    let silenceStart = Date.now();

    function checkSilence() {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;

        if (average < 10) { // Low volume means silence
            if (Date.now() - silenceStart > silenceDelay) {
                callback();
                return;
            }
        } else {
            silenceStart = Date.now(); // Reset timer if voice detected
        }

        requestAnimationFrame(checkSilence);
    }

    checkSilence();
}

navigator.mediaDevices.getUserMedia({ audio: true })
  .then((stream) => {
      console.log("Microphone access granted");
  })
  .catch((error) => {
      alert("Microphone access is required!");
      console.error("Error accessing microphone:", error);
  });

  
  // Refresh button clears the input field and response container
  document.getElementById("refreshBtn").addEventListener("click", () => {
      document.getElementById("queryInput").value = ""; // Clears input
      document.getElementById("responseContainer").innerText = ""; // Clears response
  });
});
