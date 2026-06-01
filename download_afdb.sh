#!/bin/bash
# download_afdb.sh

DEST="data/afib/raw"
TEMP_ZIP="afdb_temp.zip"
ZIP_URL="https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip"

echo "--------------------------------------------------------"
echo "Starting high-speed download of AFDB via ZIP archive..."
echo "--------------------------------------------------------"

mkdir -p "$DEST"

# Download the ZIP file (faster than individual requests)
echo "Downloading archive (~440MB)..."
if curl -L --progress-bar -o "$TEMP_ZIP" "$ZIP_URL"; then
    echo "Download successful."
    
    echo "Extracting records to $DEST..."
    # -j: junk paths (don't create subdirectories)
    # -o: overwrite existing files
    # -q: quiet
    unzip -j -o -q "$TEMP_ZIP" -d "$DEST"
    
    if [ $? -eq 0 ]; then
        echo "Extraction complete."
        rm "$TEMP_ZIP"
        echo "Cleaned up temporary files."
    else
        echo "Error: Extraction failed."
        exit 1
    fi
else
    echo "Error: Failed to download ZIP from PhysioNet."
    exit 1
fi

echo "--------------------------------------------------------"
echo "AFDB records are now available in $DEST"
echo "--------------------------------------------------------"
