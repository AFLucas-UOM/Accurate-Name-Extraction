// ========== STEP MANAGEMENT ==========
const step1El = document.getElementById('step1');
const step2El = document.getElementById('step2');
const step3El = document.getElementById('step3');
const step4El = document.getElementById('step4');
const step5El = document.getElementById('step5');

const uploadSection = document.getElementById('upload-section');
const optionsSection = document.getElementById('options-section');
const confirmSection = document.getElementById('confirm-section');
const analysisSection = document.getElementById('analysis-section');
const resultsSection = document.getElementById('results-section');

function goToStep(stepNumber) {
  [uploadSection, optionsSection, confirmSection, analysisSection, resultsSection].forEach(sec => sec.classList.remove('active'));
  [step1El, step2El, step3El, step4El, step5El].forEach(s => s.classList.remove('active'));

  switch(stepNumber) {
    case 1:
      uploadSection.classList.add('active');
      step1El.classList.add('active');
      break;
    case 2:
      optionsSection.classList.add('active');
      step2El.classList.add('active');
      break;
    case 3:
      confirmSection.classList.add('active');
      step3El.classList.add('active');
      break;
    case 4:
      analysisSection.classList.add('active');
      step4El.classList.add('active');
      simulateProgress(); // Start progress simulation in Step 4
      break;
    case 5:
      resultsSection.classList.add('active');
      step5El.classList.add('active');
      displayResults(); // Display results in Step 5
      break;
    default:
      uploadSection.classList.add('active');
      step1El.classList.add('active');
      break;
  }
}

// ========== STEP 1: UPLOAD ==========
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const errorMessage = document.getElementById('error-message');
const successMessage = document.getElementById('success-message');
const videoInfo = document.getElementById('video-info');
const videoPreviewContainer = document.getElementById('preview-container');
const videoPreview = document.getElementById('video-preview');
const nextBtn1 = document.getElementById('next-btn-1');
const uploadInstructions = document.getElementById('upload-instructions');

let selectedFile = null;
let videoDuration = null;

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, e => e.preventDefault(), false);
  document.body.addEventListener(eventName, e => e.preventDefault(), false);
});

// Highlight drop area on drag
['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, () => dropArea.classList.add('hover'), false);
});
['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, () => dropArea.classList.remove('hover'), false);
});

// Handle dropped files
dropArea.addEventListener('drop', e => {
  const dt = e.dataTransfer;
  if (dt.files.length) handleFiles(dt.files);
});

// Click to open file dialog
dropArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFiles(fileInput.files);
});

function handleFiles(files) {
  resetMessages();
  const file = files[0];
  if (!file) return;

  // Maximum file size check (50 MB)
  const maxSizeMB = 50;
  if (file.size > maxSizeMB * 1024 * 1024) {
    showError(`File size exceeds the maximum limit of ${maxSizeMB} MB.`);
    fileInput.value = "";
    return;
  }

  // Accept only allowed video formats
  const allowedTypes = ["video/mp4", "video/webm", "video/ogg", "video/quicktime", "video/x-matroska"];
  if (!allowedTypes.includes(file.type)) {
    showError(`Unsupported format: ${file.type}. Please select a standard video file.`);
    fileInput.value = "";
    return;
  }

  selectedFile = file;
  nextBtn1.disabled = false;

  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
  videoInfo.style.display = 'block';
  videoInfo.innerHTML = `
    <strong>File Name:</strong> ${file.name}<br>
    <strong>Video Duration:</strong> Loading...<br>
    <strong>Size:</strong> ${sizeMB} MB
  `;

  videoPreviewContainer.style.display = 'block';
  videoPreview.src = URL.createObjectURL(file);
  videoPreview.load();
  videoPreview.addEventListener('loadedmetadata', function() {
    videoDuration = videoPreview.duration;
    let minutes = Math.floor(videoDuration / 60);
    let seconds = Math.floor(videoDuration % 60);
    videoInfo.innerHTML = `
      <strong>File Name:</strong> ${file.name}<br>
      <strong>Video Duration:</strong> ${minutes}m ${seconds}s<br>
      <strong>Size:</strong> ${sizeMB} MB
    `;
  }, { once: true });

  // Update drop-content so that file input remains intact
  document.getElementById('drop-content').innerHTML = `
    <p class="text-success"><i class="bi bi-check-circle me-2"></i>${file.name} uploaded successfully!</p>
    <p class="mt-2">Click here or drag another video to replace the current upload.</p>
  `;
  uploadInstructions.style.display = 'block';
  showSuccess("Video uploaded successfully!");
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorMessage.style.display = 'block';
  console.error(msg);
}

function showSuccess(msg) {
  successMessage.textContent = msg;
  successMessage.style.display = 'block';
  setTimeout(() => {
    successMessage.style.display = 'none';
  }, 3000);
}

function resetMessages() {
  errorMessage.style.display = 'none';
  successMessage.style.display = 'none';
  uploadInstructions.style.display = 'none';
}

nextBtn1.addEventListener('click', () => {
  goToStep(2);
});

// ========== STEP 2: SELECT OPTIONS ==========
const backBtn2 = document.getElementById('back-btn-2');
const nextBtn2 = document.getElementById('next-btn-2');
const modelSelect = document.getElementById('model-select');

backBtn2.addEventListener('click', () => {
  goToStep(1);
});

nextBtn2.addEventListener('click', () => {
  goToStep(3);
  document.getElementById('confirm-filename').textContent = selectedFile ? selectedFile.name : 'N/A';
  document.getElementById('confirm-duration').textContent = videoDuration ?
    `${Math.floor(videoDuration / 60)}m ${Math.floor(videoDuration % 60)}s` : 'N/A';
  document.getElementById('confirm-filesize').textContent = selectedFile ?
    ((selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB') : 'N/A';
  document.getElementById('confirm-model').textContent = modelSelect.value;
});

// ========== STEP 3: CONFIRMATION ==========
const backBtn3 = document.getElementById('back-btn-3');
const runAnalysisBtn = document.getElementById('run-analysis-btn');

backBtn3.addEventListener('click', () => {
  goToStep(2);
});

runAnalysisBtn.addEventListener('click', () => {
  goToStep(4);
  simulateProgress();
});

// ========== STEP 4: ANALYSIS IN PROGRESS ==========
const cancelAnalysisBtn = document.getElementById('cancel-analysis-btn');
cancelAnalysisBtn.addEventListener('click', () => {
  const cancelModal = new bootstrap.Modal(document.getElementById('cancelModal'));
  cancelModal.show();
});

document.getElementById('confirm-cancel-btn').addEventListener('click', () => {
  window.location.reload();
});

// ========== PROGRESS BAR SIMULATION ==========
function simulateProgress() {
  const progressBar = document.getElementById('progress-bar');
  let progress = 0;
  progressBar.style.width = progress + '%';
  progressBar.textContent = progress + '%';
  const interval = setInterval(() => {
    if (progress >= 100) {
      clearInterval(interval);
      progressBar.textContent = 'Complete';
      setTimeout(() => {
        goToStep(5);
      }, 1000);
    } else {
      progress += 10;
      progressBar.style.width = progress + '%';
      progressBar.textContent = progress + '%';
    }
  }, 500);
}

// ========== STEP 5: RESULTS ==========
const resultsContainer = document.getElementById('results-container');
const downloadResultsBtn = document.getElementById('download-results-btn');

function displayResults() {
  resultsContainer.innerHTML = `<p>Process Complete. Analysis results go here.</p>`;
}

downloadResultsBtn.addEventListener('click', () => {
  const results = { message: "Analysis Complete" };
  const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'analysis_results.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

// ========== RESET FUNCTION ==========
function resetAll() {
  fileInput.value = "";
  selectedFile = null;
  videoDuration = null;
  document.getElementById('drop-content').innerHTML = `
    <i class="bi bi-upload" style="font-size: 2.5rem;"></i>
    <p class="mt-2">Drag & drop your video here, or click to select</p>
  `;
  resetMessages();
  videoInfo.style.display = 'none';
  videoPreviewContainer.style.display = 'none';
  nextBtn1.disabled = true;
}

// Initialize at Step 1
goToStep(1);
