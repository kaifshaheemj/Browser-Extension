const aiBox = document.createElement("div");
aiBox.style.position = "fixed";
aiBox.style.top = "100px";  // Adjust as per the orange box position
aiBox.style.right = "20px"; // Position near the sidebar
aiBox.style.width = "400px"; // Slightly wider for better content fit
aiBox.style.height = "550px"; // Adjusted height for full content
aiBox.style.background = "white";
aiBox.style.border = "1px solid #ccc";
aiBox.style.borderRadius = "10px";
aiBox.style.boxShadow = "0px 0px 10px rgba(0,0,0,0.2)";
aiBox.style.padding = "0"; // Remove extra padding
aiBox.style.overflow = "hidden"; // Ensure no scrollbars
aiBox.style.zIndex = "99999"; // Ensure it's above everything

// Load the popup content inside
aiBox.innerHTML = `
    <iframe 
        src="${chrome.runtime.getURL("popup.html")}" 
        width="100%" 
        height="100%" 
        frameborder="0"
        style="display: block; border-radius: 10px;">
    </iframe>`;

// Add to the page
document.body.appendChild(aiBox);
