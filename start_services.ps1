# Create and activate Python virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Green
python -m venv venv
./venv/Scripts/Activate

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Green
pip install -r requirements_camera.txt
pip install -r fauna-rag/requirements.txt
pip install flask flask-cors pillow onnxruntime numpy pandas

# Check for required model and data files
Write-Host "Checking for required files..." -ForegroundColor Green

# Check for ONNX model
if (-not (Test-Path "ImageClassifier.onnx")) {
    Write-Host "❌ ImageClassifier.onnx not found!" -ForegroundColor Red
    Write-Host "Please make sure the ONNX model file is in the root directory." -ForegroundColor Yellow
    exit 1
}

# Check for species mapping
if (-not (Test-Path "official_species_mapping.json")) {
    Write-Host "Species mapping not found. Attempting to create it..." -ForegroundColor Yellow
    
    # Look for CSV file
    $csvFiles = Get-ChildItem -Filter "*.csv"
    if ($csvFiles.Count -eq 0) {
        Write-Host "❌ No CSV files found for species mapping!" -ForegroundColor Red
        Write-Host "Please provide a CSV file with species data." -ForegroundColor Yellow
        exit 1
    }
    
    # Use the first CSV file found
    Write-Host "Found CSV file: $($csvFiles[0].Name)" -ForegroundColor Green
    python load_species_csv.py
    
    if (-not (Test-Path "official_species_mapping.json")) {
        Write-Host "❌ Failed to create species mapping!" -ForegroundColor Red
        exit 1
    }
}

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Green
Set-Location -Path ai-survival-guide-frontend
npm install --yes
Set-Location -Path ..

# Function to start a Python script in a new window
function Start-PythonScript {
    param(
        [string]$ScriptPath,
        [string]$WindowTitle
    )
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        cd '$pwd';
        ./venv/Scripts/Activate;
        Write-Host 'Starting $WindowTitle...' -ForegroundColor Green;
        python '$ScriptPath'
    " -WindowStyle Normal
}

# Start all services
Write-Host "Starting all services..." -ForegroundColor Green

# Start backend services
Start-PythonScript -ScriptPath "api_server.py" -WindowTitle "API Server"
Start-PythonScript -ScriptPath "fauna-rag/app.py" -WindowTitle "Fauna RAG Service"
Start-PythonScript -ScriptPath "modern_analyzer.py" -WindowTitle "Modern Analyzer"

# Start frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    cd '$pwd/ai-survival-guide-frontend';
    Write-Host 'Starting frontend...' -ForegroundColor Green;
    npm start
" -WindowStyle Normal

Write-Host "All services have been started!" -ForegroundColor Green
Write-Host "Press Ctrl+C in each window to stop the services." -ForegroundColor Yellow 