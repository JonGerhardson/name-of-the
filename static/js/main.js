// static/js/main.js

// --- Get DOM Elements (do this once at the top) ---
const form = document.getElementById('upload-form');
const uploadButton = document.getElementById('upload-button');
const statusDiv = document.getElementById('status');
const statusMessage = document.getElementById('status-message');
const progressBar = document.getElementById('progress-bar');
const taskIdDisplay = document.getElementById('task-id-display');
const enrollmentSection = document.getElementById('enrollment-section');
const enrollmentForm = document.getElementById('enrollment-form');
const enrollmentItemsDiv = document.getElementById('enrollment-items'); // Should be inside the form now
const enrollmentSubmitButton = document.getElementById('enrollment-submit-button');
const enrollmentAudioPlayer = document.getElementById('enrollment-audio-player');
const enrollmentAudioSource = document.getElementById('enrollment-audio-source'); // Get source element if using <source> tag

const resultsDiv = document.getElementById('results');
const resultContent = document.getElementById('result-content');
const downloadLinkP = document.getElementById('download-link-p');
const downloadLink = document.getElementById('download-link');
const downloadJsonLinkP = document.getElementById('download-json-link-p');
const downloadJsonLink = document.getElementById('download-json-link');
const downloadLogLinkP = document.getElementById('download-log-link-p');
const downloadLogLink = document.getElementById('download-log-link');

// --- Global Variables ---
let currentTaskId = null;
let currentJobId = null;
let pollingInterval = null;
let stage1ResultData = null;
let enrollmentAudioTimeout = null; // To store timeout ID for stopping playback

// --- Utility Functions ---
function updateStatusDisplay(message, state = 'INFO', progress = null) {
    if (!statusDiv || !statusMessage || !progressBar || !taskIdDisplay) {
        console.error("[updateStatusDisplay] One or more status elements not found!");
        return;
    }
    statusDiv.classList.remove('hidden');
    statusMessage.textContent = message;
    statusMessage.className = 'status-message'; // Reset classes

    let alertClass = 'alert-info'; // Default
    if (state === 'FAILURE' || state === 'ERROR') {
        statusMessage.classList.add('error-message');
        alertClass = 'alert-danger';
    } else if (state === 'SUCCESS') {
        statusMessage.classList.add('success-message');
        alertClass = 'alert-success';
    } else if (state === 'AWAITING_ENROLLMENT') {
        statusMessage.classList.add('warning-message');
        alertClass = 'alert-warning';
    }
    statusDiv.className = `alert ${alertClass}`; // Set alert class

    taskIdDisplay.textContent = currentTaskId ? `Task ID: ${currentTaskId}` : '';

    if ((state === 'PROGRESS' && progress !== null) || state === 'SUCCESS') {
        progressBar.classList.remove('hidden');
        progressBar.value = (state === 'SUCCESS') ? 100 : progress;
    } else {
        progressBar.classList.add('hidden');
        progressBar.value = 0;
    }
}

function hideStatus() {
    if (statusDiv) statusDiv.classList.add('hidden');
    if (progressBar) progressBar.classList.add('hidden');
    if (taskIdDisplay) taskIdDisplay.textContent = '';
}

function resetUI() {
    console.log("[resetUI] Resetting UI elements.");
    hideStatus();
    if (enrollmentSection) enrollmentSection.classList.add('hidden');
    if (enrollmentItemsDiv) enrollmentItemsDiv.innerHTML = '';
    // --- NEW: Reset shared audio player ---
    if (enrollmentAudioPlayer) {
        enrollmentAudioPlayer.pause();
        if (enrollmentAudioSource) {
             enrollmentAudioSource.removeAttribute('src'); // Remove src from source tag
        } else {
             enrollmentAudioPlayer.removeAttribute('src'); // Remove src from audio tag directly
        }
        // Check if load() exists before calling
        if (typeof enrollmentAudioPlayer.load === 'function') {
            enrollmentAudioPlayer.load(); // Reload to apply src removal
        }
        enrollmentAudioPlayer.classList.add('hidden'); // Hide player
    }
    if (enrollmentAudioTimeout) clearTimeout(enrollmentAudioTimeout); // Clear any pending playback stop

    if (resultsDiv) resultsDiv.classList.add('hidden');
    if (resultContent) resultContent.innerHTML = '';
    if (downloadLinkP) downloadLinkP.classList.add('hidden');
    if (downloadJsonLinkP) downloadJsonLinkP.classList.add('hidden');
    if (downloadLogLinkP) downloadLogLinkP.classList.add('hidden');
    if (uploadButton) uploadButton.disabled = false;
    if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = false;

    stage1ResultData = null;
    currentJobId = null;
    currentTaskId = null;

    if (pollingInterval) {
        console.log("[resetUI] Clearing polling interval:", pollingInterval);
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
    if (window.pollingIntervalId) {
        console.log("[resetUI] Clearing window.pollingIntervalId:", window.pollingIntervalId);
        clearInterval(window.pollingIntervalId);
        window.pollingIntervalId = null;
    }
}

// --- Event Listeners ---
if (form) {
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        console.log("[form.submit] Form submitted.");
        resetUI();

        const formData = new FormData(form);
        const fileInput = document.getElementById('file');

        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            updateStatusDisplay('Error: Please select a file.', 'ERROR');
            return;
        }
        updateStatusDisplay('Uploading file...', 'INFO', 5);
        if (uploadButton) uploadButton.disabled = true;

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (response.ok && data.task_id) {
                currentJobId = data.job_id;
                currentTaskId = data.task_id;
                console.log(`[form.submit] Upload successful. Job ID: ${currentJobId}, Task ID: ${currentTaskId}`);
                updateStatusDisplay('Processing started (Stage 1)...', 'INFO', 10);
                startPolling(currentTaskId);
            } else {
                throw new Error(data.error || `Server error: ${response.status}`);
            }
        } catch (error) {
            console.error('[form.submit] Upload error:', error);
            updateStatusDisplay(`Upload failed: ${error.message}`, 'ERROR');
            if (uploadButton) uploadButton.disabled = false;
        }
    });
} else {
    console.error("Upload form not found!");
}

if (enrollmentForm) {
    enrollmentForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        console.log("[enrollmentForm.submit] Enrollment submitted.");
        if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = true;
        updateStatusDisplay('Submitting enrollment and starting finalization...', 'INFO', 50);

        const enrollmentMap = {};
        if (enrollmentItemsDiv) {
            const inputs = enrollmentItemsDiv.querySelectorAll('input[type="text"]');
            inputs.forEach(input => {
                const tempId = input.dataset.tempId;
                const enteredName = input.value.trim();
                if (enteredName && tempId) {
                    enrollmentMap[tempId] = enteredName;
                }
            });
        }
        console.log("[enrollmentForm.submit] Enrollment map created:", enrollmentMap);

        // --- Check required data (output_dir and original_kwargs still needed from stage 1 result) ---
        if (!stage1ResultData || !stage1ResultData.output_dir || !stage1ResultData.original_kwargs || !currentJobId) {
            console.error("[enrollmentForm.submit] Enrollment Submit Error: Missing stage1ResultData or currentJobId", stage1ResultData, currentJobId);
            updateStatusDisplay('Error: Missing necessary data from previous stage. Cannot proceed.', 'ERROR');
            if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = false;
            return;
        }

        const payload = {
            job_id: currentJobId,
            output_dir: stage1ResultData.output_dir,
            enrollment_map: enrollmentMap,
            original_kwargs: stage1ResultData.original_kwargs
        };
        console.log("[enrollmentForm.submit] Submitting enrollment payload:", JSON.stringify(payload));

        try {
            const response = await fetch('/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();

            if (response.ok && data.finalize_task_id) {
                if (enrollmentSection) enrollmentSection.classList.add('hidden');
                // --- NEW: Reset shared audio player after submission ---
                 if (enrollmentAudioPlayer) {
                    enrollmentAudioPlayer.pause();
                    if (enrollmentAudioSource) {
                         enrollmentAudioSource.removeAttribute('src');
                    } else {
                         enrollmentAudioPlayer.removeAttribute('src');
                    }
                    if (typeof enrollmentAudioPlayer.load === 'function') {
                        enrollmentAudioPlayer.load();
                    }
                    enrollmentAudioPlayer.classList.add('hidden');
                }
                if (enrollmentAudioTimeout) clearTimeout(enrollmentAudioTimeout);

                currentTaskId = data.finalize_task_id;
                console.log(`[enrollmentForm.submit] Enrollment submitted. Finalization Task ID: ${currentTaskId}`);
                updateStatusDisplay('Enrollment submitted. Finalizing process...', 'INFO', 60);
                startPolling(currentTaskId);
            } else {
                throw new Error(data.error || `Server error: ${response.status}`);
            }
        } catch (error) {
            console.error('[enrollmentForm.submit] Enrollment submission error:', error);
            updateStatusDisplay(`Enrollment submission failed: ${error.message}`, 'ERROR');
            if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = false;
        }
    });
} else {
    console.error("Enrollment form not found!");
}


// --- Status Polling ---
function startPolling(taskId) {
    if (pollingInterval) {
        console.log(`[startPolling] Clearing previous interval ${pollingInterval} before starting new one for ${taskId}`);
        clearInterval(pollingInterval);
    }
    if (window.pollingIntervalId) {
        console.log(`[startPolling] Clearing previous window.pollingIntervalId ${window.pollingIntervalId}`);
        clearInterval(window.pollingIntervalId);
    }

    console.log("[startPolling] Starting polling for task:", taskId);
    window.pollingIntervalId = setInterval(() => checkStatus(taskId), 2000);
    pollingInterval = window.pollingIntervalId;
}

async function checkStatus(taskId) {
     if (taskId !== currentTaskId) {
         console.log(`[checkStatus] Polling stopped for old task ${taskId}. Current task is ${currentTaskId}.`);
         if (window.pollingIntervalId) {
             clearInterval(window.pollingIntervalId);
             window.pollingIntervalId = null;
             pollingInterval = null;
         }
         return;
     }
     // Check both interval references before proceeding
     if (!pollingInterval && !window.pollingIntervalId) {
         console.log("[checkStatus] Polling interval reference is null, stopping check for", taskId);
         return
     }

    console.log(`[checkStatus] Checking status for task ${taskId}...`);
    try {
        const response = await fetch(`/status/${taskId}`);
        if (!response.ok) {
             throw new Error(`Server error fetching status: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        console.log("[checkStatus] Status response data:", data);

        // Add extra check: ensure the task ID from the response matches the one we are polling
        // This helps if the server somehow returns status for a different task ID on the same endpoint call
        if (data.task_id !== taskId) {
             console.warn(`[checkStatus] Received status for task ${data.task_id}, but currently polling for ${taskId}. Ignoring.`);
             return;
        }
        // Also ensure it matches the globally stored current task ID
        if (data.task_id !== currentTaskId) {
            console.log(`[checkStatus] Ignoring status update for task ${data.task_id}. Current task is ${currentTaskId}`);
            return;
        }


        const state = data.state;
        const result = data.result; // This will contain the structure with timestamps
        const statusText = data.status || state;
        const progress = data.progress;
        const errorText = data.error;

        updateStatusDisplay(statusText, state, progress);

        // --- State Machine Logic ---
        const isPollingActive = pollingInterval || window.pollingIntervalId;

        if (state === 'SUCCESS' || state === 'FAILURE' || state === 'AWAITING_ENROLLMENT') {
             if (isPollingActive) {
                 console.log(`[checkStatus] Task ${taskId} reached terminal state ${state}. Clearing interval.`);
                 clearInterval(pollingInterval); pollingInterval = null; window.pollingIntervalId = null;
             }
             if (uploadButton) uploadButton.disabled = false; // Re-enable upload on terminal state
        }

        if (state === 'PROGRESS' || state === 'PENDING' || state === 'STARTED') {
            if (enrollmentSection) enrollmentSection.classList.add('hidden');
            if (resultsDiv) resultsDiv.classList.add('hidden');
        } else if (state === 'SUCCESS') {
             // Check the actual result payload status
            if (result && result.status === 'enrollment_required') {
                console.log("[checkStatus] Task SUCCESS -> enrollment_required.");
                hideStatus();
                if (resultsDiv) resultsDiv.classList.add('hidden');
                if (enrollmentSection) enrollmentSection.classList.remove('hidden');
                stage1ResultData = result; // Store the whole result including timestamps and original audio path
                // --- Pass the result data to the display function ---
                displayEnrollmentForm(result);
            } else if (result && result.final_transcript_filename) {
                console.log("[checkStatus] Finalization task SUCCESS -> Displaying results.");
                hideStatus();
                if (enrollmentSection) enrollmentSection.classList.add('hidden');
                displayResults(result); // Pass the full result object
            } else {
                console.log("[checkStatus] Generic SUCCESS (Stage 1 no enrollment).");
                hideStatus();
                if (enrollmentSection) enrollmentSection.classList.add('hidden');
                if (resultsDiv) resultsDiv.classList.remove('hidden');
                const successMessage = (result && result.message) ? result.message : 'Task completed successfully.';
                if(resultContent) resultContent.innerHTML = `<p class="alert alert-success">${successMessage}</p>`;
                 if (result && result.temp_transcript_path && currentJobId) {
                     console.log("[checkStatus] Intermediate files might be available:", result.temp_transcript_path);
                 }
            }
        } else if (state === 'AWAITING_ENROLLMENT') {
            // This state might still be used by Celery itself, handle similarly to SUCCESS + enrollment_required
            console.log(`[checkStatus] Task ${taskId} AWAITING_ENROLLMENT.`);
            // Interval cleared above
            hideStatus();
            if (resultsDiv) resultsDiv.classList.add('hidden');
            if (enrollmentSection) enrollmentSection.classList.remove('hidden');
            stage1ResultData = result; // Store result (should contain timestamps etc.)
            console.log("[checkStatus] Calling displayEnrollmentForm for AWAITING_ENROLLMENT.");
            displayEnrollmentForm(result); // Pass the full result object
        } else if (state === 'FAILURE') {
            console.log(`[checkStatus] Task ${taskId} FAILURE.`);
            // Interval cleared above
            if (enrollmentSection) enrollmentSection.classList.add('hidden');
            if (resultsDiv) resultsDiv.classList.add('hidden');
            updateStatusDisplay(`Task Failed: ${errorText || statusText || 'Unknown error'}`, 'FAILURE');
        } else {
             console.warn("[checkStatus] Unhandled task state:", state);
        }

    } catch (error) {
        console.error('[checkStatus] Status check fetch/processing error:', error);
        updateStatusDisplay(`Error checking status: ${error.message}`, 'ERROR');
        if (pollingInterval) clearInterval(pollingInterval);
        if (window.pollingIntervalId) clearInterval(window.pollingIntervalId);
        pollingInterval = null; window.pollingIntervalId = null;
        if (uploadButton) uploadButton.disabled = false;
    }
}

// --- UI Display Functions ---

/**
 * Displays the enrollment form, now using timestamp-based playback and showing text snippets.
 * @param {object} stage1Data - The result object from stage 1, containing
 * unknown_speakers array, original_audio_path, and text_snippet for each speaker.
 */
function displayEnrollmentForm(stage1Data) {
    console.log("[displayEnrollmentForm] Called with stage 1 data:", stage1Data);

    // --- Destructure necessary data ---
    const unknownSpeakers = Array.isArray(stage1Data?.unknown_speakers) ? stage1Data.unknown_speakers : [];
    const originalAudioRelativePath = stage1Data?.original_audio_path;

    // --- Check if essential elements exist ---
    if (!enrollmentItemsDiv || !enrollmentAudioPlayer) {
        console.error("[displayEnrollmentForm] CRITICAL: enrollmentItemsDiv or enrollmentAudioPlayer element not found!");
        updateStatusDisplay("Internal UI Error: Cannot display enrollment form.", "ERROR");
        return;
    }
    if (!enrollmentSubmitButton) console.error("[displayEnrollmentForm] CRITICAL: enrollmentSubmitButton element not found!");
    if (!enrollmentSection) console.error("[displayEnrollmentForm] CRITICAL: enrollmentSection element not found!");

    console.log("[displayEnrollmentForm] Target enrollmentItemsDiv:", enrollmentItemsDiv);
    console.log("[displayEnrollmentForm] Current Job ID:", currentJobId);
    console.log("[displayEnrollmentForm] Original Audio Relative Path:", originalAudioRelativePath);

    enrollmentItemsDiv.innerHTML = ''; // Clear previous items

    // --- Validate necessary data for playback ---
    if (!currentJobId) {
        console.error("[displayEnrollmentForm] CRITICAL: currentJobId is null or undefined! Cannot create audio URL.");
        enrollmentItemsDiv.innerHTML = '<p class="error-message">Internal Error: Cannot load audio player (missing job identifier).</p>';
        if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = true;
        if (enrollmentSection) enrollmentSection.classList.remove('hidden');
        return;
    }
    if (!originalAudioRelativePath) {
        console.error("[displayEnrollmentForm] CRITICAL: original_audio_path missing from stage 1 result!");
        enrollmentItemsDiv.innerHTML = '<p class="error-message">Internal Error: Cannot load audio player (missing audio file path).</p>';
        if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = true;
        if (enrollmentSection) enrollmentSection.classList.remove('hidden');
        return;
    }

    // --- Setup the shared audio player ---
    const originalAudioUrl = `/results/${currentJobId}/${originalAudioRelativePath}`;
    console.log("[displayEnrollmentForm] Setting shared audio player source:", originalAudioUrl);
    if (enrollmentAudioSource) {
        enrollmentAudioSource.src = originalAudioUrl;
    } else {
        enrollmentAudioPlayer.src = originalAudioUrl;
    }
    if (typeof enrollmentAudioPlayer.load === 'function') {
        enrollmentAudioPlayer.load(); // Load the new source
    }
    enrollmentAudioPlayer.classList.remove('hidden'); // Show the player

    // Handle potential loading errors for the main audio
    enrollmentAudioPlayer.onerror = (e) => {
         console.error("[displayEnrollmentForm] CRITICAL: Error loading main audio file:", originalAudioUrl, e);
         // Display a more specific error in the enrollment items div
         enrollmentItemsDiv.innerHTML = `<p class="error-message">Error loading main audio file from ${originalAudioUrl}. Cannot play samples. Check server logs and file path.</p>`;
         if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = true;
         enrollmentAudioPlayer.classList.add('hidden');
    };
     enrollmentAudioPlayer.oncanplay = () => {
        console.log("[displayEnrollmentForm] Main audio file ready for playback.");
    };


    // --- Populate speaker items ---
    console.log("[displayEnrollmentForm] Checking unknownSpeakers:", unknownSpeakers);
    console.log("[displayEnrollmentForm] Number of unknown speakers:", unknownSpeakers ? unknownSpeakers.length : 'null/undefined');

    if (!unknownSpeakers || unknownSpeakers.length === 0) {
        console.log("[displayEnrollmentForm] Condition evaluated to true: No speakers or empty array.");
        enrollmentItemsDiv.innerHTML = '<p>No unknown speakers detected.</p>';
        if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = true;
        console.log("[displayEnrollmentForm] No unknown speakers to display.");
        if (enrollmentSection) enrollmentSection.classList.remove('hidden'); // Ensure section is visible to show message
        return;
    } else {
         console.log("[displayEnrollmentForm] Condition evaluated to false: Processing speakers.");
    }


    if (enrollmentSubmitButton) enrollmentSubmitButton.disabled = false;
    if (enrollmentSection) enrollmentSection.classList.remove('hidden');

    console.log("[displayEnrollmentForm] Proceeding to populate form items...");

    unknownSpeakers.forEach((speaker, index) => {
        // --- **MODIFIED:** Validate speaker data (text_snippet is now optional) ---
        if (!speaker || typeof speaker !== 'object' || !speaker.temp_id ||
            typeof speaker.start_time !== 'number' || typeof speaker.end_time !== 'number') {
            console.warn(`[displayEnrollmentForm] Skipping invalid speaker data (missing required fields) at index ${index}:`, speaker);
            const errorItemDiv = document.createElement('div');
            errorItemDiv.classList.add('enrollment-item', 'error-message');
            errorItemDiv.textContent = `Error: Invalid data received for speaker at index ${index}. Cannot create enrollment item.`;
            if (enrollmentItemsDiv) enrollmentItemsDiv.appendChild(errorItemDiv);
            return; // continue to next speaker
        }

        const tempId = speaker.temp_id;
        const originalLabel = speaker.original_label || 'N/A';
        const startTime = speaker.start_time;
        const endTime = speaker.end_time;
        // **MODIFIED:** Get text snippet or use a default if missing/not a string
        const textSnippet = (typeof speaker.text_snippet === 'string' && speaker.text_snippet)
                            ? speaker.text_snippet
                            : "(Text preview unavailable)";

        console.log(`[displayEnrollmentForm] Creating item ${index + 1}: tempId=${tempId}, start=${startTime}, end=${endTime}`);

        const itemDiv = document.createElement('div');
        itemDiv.classList.add('enrollment-item');
        itemDiv.id = `enrollment-item-${tempId}`;

        const label = document.createElement('label');
        label.htmlFor = `enroll-${tempId}`;
        label.textContent = `Enter name for ${tempId} (Original: ${originalLabel}):`;

        const input = document.createElement('input');
        input.type = 'text';
        input.id = `enroll-${tempId}`;
        input.name = tempId;
        input.dataset.tempId = tempId; // Store tempId for submission
        input.placeholder = "Leave blank to skip";

        // --- Create Play Button ---
        const playButton = document.createElement('button');
        playButton.type = 'button'; // Prevent form submission
        playButton.textContent = 'Play Sample';
        playButton.classList.add('play-sample-button'); // Add class for styling
        playButton.dataset.startTime = startTime; // Store times on the button
        playButton.dataset.endTime = endTime;

        playButton.addEventListener('click', () => {
            playAudioSegment(parseFloat(playButton.dataset.startTime), parseFloat(playButton.dataset.endTime));
        });

        // --- Create Text Snippet Display ---
        const textSnippetPara = document.createElement('p');
        textSnippetPara.style.fontStyle = 'italic';
        textSnippetPara.style.marginTop = '5px';
        textSnippetPara.style.paddingLeft = '10px';
        textSnippetPara.style.borderLeft = '3px solid #ccc';
        textSnippetPara.textContent = `"${textSnippet}"`; // Display text in quotes

        // --- Append elements to itemDiv ---
        const labelContainer = document.createElement('div'); // Container for label and button
        labelContainer.appendChild(label);
        labelContainer.appendChild(playButton); // Button next to label

        itemDiv.appendChild(labelContainer); // Add label+button container
        itemDiv.appendChild(textSnippetPara); // Add text snippet paragraph
        itemDiv.appendChild(input); // Add input below

        // --- Append itemDiv to main container ---
        if (enrollmentItemsDiv) {
            enrollmentItemsDiv.appendChild(itemDiv);
            console.log(`[displayEnrollmentForm] Appended item for ${tempId} to enrollmentItemsDiv.`);
        } else {
            console.error(`[displayEnrollmentForm] Tried to append item for ${tempId}, but enrollmentItemsDiv was null!`);
        }
    });

    // Final check
    if (enrollmentItemsDiv && enrollmentItemsDiv.children.length > 0) {
        console.log(`[displayEnrollmentForm] Successfully populated enrollment items. Child count: ${enrollmentItemsDiv.children.length}`);
    } else if (unknownSpeakers.length > 0) {
        console.error("[displayEnrollmentForm] Loop finished, but enrollmentItemsDiv has no children. Check speaker data or if element was removed.");
        if (enrollmentItemsDiv) enrollmentItemsDiv.innerHTML = '<p class="error-message">Error displaying enrollment form items.</p>';
    } else {
        // This case should now be handled by the initial check
    }
}

/**
 * Plays a segment of the shared enrollment audio player.
 * @param {number} startTime - The time in seconds to start playback.
 * @param {number} endTime - The time in seconds to stop playback.
 */
function playAudioSegment(startTime, endTime) {
    if (!enrollmentAudioPlayer || !enrollmentAudioPlayer.src || enrollmentAudioPlayer.readyState < 2) { // readyState < 2 means not enough data
        console.error("[playAudioSegment] Audio player not ready or no source set.");
        // Use a less intrusive notification if possible, or just log
        console.warn("Audio player not ready to play segment yet.");
        return;
    }

    // Clear any previous timeout used to stop playback
    if (enrollmentAudioTimeout) {
        clearTimeout(enrollmentAudioTimeout);
        enrollmentAudioTimeout = null;
    }

    // Ensure start/end times are valid
    const audioDuration = enrollmentAudioPlayer.duration;
    if (isNaN(startTime) || isNaN(endTime) || startTime < 0 || endTime <= startTime || endTime > audioDuration) {
        console.error(`[playAudioSegment] Invalid start/end times: start=${startTime}, end=${endTime}, duration=${audioDuration}`);
        console.warn("Invalid time range requested for audio sample.");
        return;
    }

    console.log(`[playAudioSegment] Playing from ${startTime.toFixed(2)}s to ${endTime.toFixed(2)}s`);

    // Seek to start time
    enrollmentAudioPlayer.currentTime = startTime;

    // Play the audio
    const playPromise = enrollmentAudioPlayer.play();

    if (playPromise !== undefined) {
        playPromise.then(_ => {
            // Playback started successfully.
            // Calculate duration and set timeout to stop
            // Ensure duration is positive
            const duration = Math.max(0, (endTime - startTime) * 1000); // Duration in milliseconds
            console.log(`[playAudioSegment] Setting timeout for ${duration}ms`);
            enrollmentAudioTimeout = setTimeout(() => {
                if (enrollmentAudioPlayer) {
                    enrollmentAudioPlayer.pause();
                    console.log(`[playAudioSegment] Playback stopped automatically at ${endTime.toFixed(2)}s`);
                }
                enrollmentAudioTimeout = null; // Clear timeout ID
            }, duration);
        }).catch(error => {
            console.error("[playAudioSegment] Error trying to play audio:", error);
            console.warn("Could not play audio sample:", error.message);
        });
    }
}


function displayResults(result) {
    console.log("[displayResults] Called with result:", result);
    if (!resultsDiv || !resultContent) {
        console.error("[displayResults] Results container or content element not found!");
        return;
    }
    resultsDiv.classList.remove('hidden');
    resultContent.innerHTML = `<p class="alert alert-success">${result.message || 'Processing finished.'}</p>`;

    if (!currentJobId) {
         console.error("[displayResults] CRITICAL: currentJobId is not set! Cannot create result URLs.");
         resultContent.innerHTML += `<p class="alert alert-warning">Could not generate download links (missing Job ID).</p>`;
         return;
    }
    // --- Construct base URL using the *current* job ID ---
    const baseUrl = `/results/${currentJobId}/`;
    console.log("[displayResults] Base URL for results:", baseUrl);

    // --- Clear/Hide previous links ---
    if (downloadLinkP) downloadLinkP.classList.add('hidden'); else console.warn("downloadLinkP not found");
    if (downloadJsonLinkP) downloadJsonLinkP.classList.add('hidden'); else console.warn("downloadJsonLinkP not found");
    if (downloadLogLinkP) downloadLogLinkP.classList.add('hidden'); else console.warn("downloadLogLinkP not found");

    // --- Populate Transcript Link ---
    if (result.final_transcript_filename && downloadLink && downloadLinkP) {
        const transcriptUrl = baseUrl + result.final_transcript_filename;
        console.log("[displayResults] Setting transcript link href:", transcriptUrl);
        downloadLink.href = transcriptUrl;
        downloadLink.textContent = `View/Download Final Transcript (${result.final_transcript_filename})`;
        downloadLinkP.classList.remove('hidden');
        // Fetch and display preview
        fetch(transcriptUrl)
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                         throw new Error(`Failed to fetch preview: ${response.status} ${response.statusText}. Server response: ${text}`);
                    });
                }
                return response.text();
             })
            .then(text => {
                const preview = document.createElement('pre');
                preview.textContent = text.substring(0, 500) + (text.length > 500 ? '...' : '');
                resultContent.appendChild(document.createElement('h3')).textContent = "Transcript Preview:";
                resultContent.appendChild(preview);
            })
            .catch(err => {
                console.error("[displayResults] Could not fetch transcript preview:", err);
                const errorMsg = document.createElement('p');
                errorMsg.textContent = `(Could not load transcript preview: ${err.message})`;
                errorMsg.style.fontStyle = 'italic';
                errorMsg.style.color = 'grey';
                resultContent.appendChild(errorMsg);
             });
    } else {
         console.warn("[displayResults] No final_transcript_filename in result or link elements missing.");
    }

    // --- Populate JSON Link ---
    if (result.intermediate_json_filename && downloadJsonLink && downloadJsonLinkP) {
        const jsonUrl = baseUrl + result.intermediate_json_filename;
        console.log("[displayResults] Setting JSON link href:", jsonUrl);
        downloadJsonLink.href = jsonUrl;
        downloadJsonLink.textContent = `View/Download Intermediate JSON (${result.intermediate_json_filename})`;
        downloadJsonLinkP.classList.remove('hidden');
    } else {
         console.warn("[displayResults] No intermediate_json_filename in result or link elements missing.");
    }

    // --- Populate Log Link ---
    if (result.log_filename && downloadLogLink && downloadLogLinkP) {
        const logUrl = baseUrl + result.log_filename;
        console.log("[displayResults] Setting Log link href:", logUrl);
        downloadLogLink.href = logUrl;
        downloadLogLink.textContent = `Download Log File (${result.log_filename})`;
        downloadLogLinkP.classList.remove('hidden');
    } else {
         console.warn("[displayResults] No log_filename in result or link elements missing.");
    }
}

// --- Initial State ---
// Add a check to ensure all main elements exist on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed.");
    // Check essential elements
    if (!form) console.error("Initialization Error: Upload form not found!");
    if (!enrollmentItemsDiv) console.error("Initialization Error: Enrollment items div not found!");
    if (!resultsDiv) console.error("Initialization Error: Results div not found!");
    // --- NEW: Check for shared audio player ---
    if (!enrollmentAudioPlayer) console.error("Initialization Error: Enrollment audio player not found! Add <audio id='enrollment-audio-player' class='hidden' controls></audio> to index.html");

    resetUI(); // Ensure clean state on initial load
});

