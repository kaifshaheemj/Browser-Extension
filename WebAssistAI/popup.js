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

  // Refresh button clears the input field and response container
  document.getElementById("refreshBtn").addEventListener("click", () => {
      document.getElementById("queryInput").value = ""; // Clears input
      document.getElementById("responseContainer").innerText = ""; // Clears response
  });
});
