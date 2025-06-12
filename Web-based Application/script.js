// Initialize the map centered on Florida (static, no interaction)
const map = L.map('map', {
    center: [27.7663, -82.6404], // Center of Florida
    zoom: 7,
    minZoom: 7,
    maxZoom: 7,
    zoomControl: false,        // Disable zoom controls
    doubleClickZoom: false,    // Disable double-click zoom
    scrollWheelZoom: false,    // Disable scroll wheel zoom
    boxZoom: false,            // Disable box zoom
    keyboard: false,           // Disable keyboard navigation
    dragging: false,           // Disable dragging/panning
    touchZoom: false,          // Disable touch zoom
    zoomSnap: 0,              // Disable zoom snapping
    zoomDelta: 0              // Disable zoom delta
});

// Define different tile layers
const streetMap = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '' // Remove attribution
});

const satelliteMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18,
    attribution: '' // Remove attribution
});

const terrainMap = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    maxZoom: 17,
    attribution: '' // Remove attribution
});

// Add default layer
streetMap.addTo(map);

// Layer control
const baseMaps = {
    "Street Map": streetMap,
    "Satellite": satelliteMap,
    "Terrain": terrainMap
};

L.control.layers(baseMaps).addTo(map);

// Set strict bounds to display only Florida
map.setMaxBounds([
    [24.0, -87.5], // Southwest boundary (includes Florida Panhandle)
    [31.5, -79.5]   // Northeast boundary (includes all of Florida)
]);

// Fit the view to Florida bounds
map.fitBounds([
    [24.5, -87.0], // Southwest 
    [31.0, -80.0]  // Northeast
]);

// Global variables for data management
let currentDataLayer = null;
let currentData = null;
let timeColumns = [];
let dataFiles = {
    'Temperature': [
        { value: '21', label: 'Dew Point Temperature (70000 Pa)' },
        { value: '64', label: 'Surface Temperature' },
        { value: '71', label: '2m Temperature' }
    ],
    'Hydrology': [
        { value: '88', label: 'Storm Surface Runoff' },
        { value: '89', label: 'Baseflow-Groundwater Runoff' }
    ],
    'Radiation': [
        { value: '170', label: 'GOES 12 Ch 3 Brightness Temperature' },
        { value: '171', label: 'GOES 12 Ch 4 Brightness Temperature' },
        { value: '172', label: 'GOES 11 Ch 3 Brightness Temperature' },
        { value: '173', label: 'GOES 11 Ch 4 Brightness Temperature' }
    ],
    'Moisture': [
        { value: '66', label: 'Moisture Availability' },
        { value: '67', label: 'Plant Canopy Surface Water' },
        { value: '83', label: 'Precipitation Rate' },
        { value: '84', label: 'Total Precipitation' },
        { value: '143', label: 'Relative Humidity' }
    ],
    'Cloud': [
        { value: '115', label: 'Total Cloud Cover' },
        { value: '116', label: 'Low Cloud Cover' },
        { value: '117', label: 'Medium Cloud Cover' },
        { value: '118', label: 'High Cloud Cover' }
    ],
    'Mass': [
        { value: '122', label: 'Pressure (Cloud Base)' },
        { value: '123', label: 'Pressure (Cloud Top)' }
    ],
    'Momentum': [
        { value: '9', label: 'Wind Speed (Gust)' },
        { value: '10', label: 'U Component of Wind (25000 Pa)' },
        { value: '11', label: 'V Component of Wind (25000 Pa)' }
    ]
};

// Data descriptions for units
const dataDescriptions = {
    // Temperature
    '21': { unit: 'K', name: 'Dew Point Temperature' },
    '64': { unit: 'K', name: 'Surface Temperature' },
    '71': { unit: 'K', name: '2m Temperature' },
    // Hydrology
    '88': { unit: 'kg m⁻²', name: 'Storm Surface Runoff' },
    '89': { unit: 'kg m⁻²', name: 'Baseflow-Groundwater Runoff' },
    // Radiation
    '170': { unit: 'K', name: 'GOES 12 Ch 3 Brightness Temp' },
    '171': { unit: 'K', name: 'GOES 12 Ch 4 Brightness Temp' },
    '172': { unit: 'K', name: 'GOES 11 Ch 3 Brightness Temp' },
    '173': { unit: 'K', name: 'GOES 11 Ch 4 Brightness Temp' },
    // Moisture
    '66': { unit: '%', name: 'Moisture Availability' },
    '67': { unit: 'kg m⁻²', name: 'Plant Canopy Surface Water' },
    '83': { unit: 'kg m⁻² s⁻¹', name: 'Precipitation Rate' },
    '84': { unit: 'kg m⁻²', name: 'Total Precipitation' },
    '143': { unit: '%', name: 'Relative Humidity' },
    // Cloud
    '115': { unit: '%', name: 'Total Cloud Cover' },
    '116': { unit: '%', name: 'Low Cloud Cover' },
    '117': { unit: '%', name: 'Medium Cloud Cover' },
    '118': { unit: '%', name: 'High Cloud Cover' },
    // Mass
    '122': { unit: 'Pa', name: 'Pressure (Cloud Base)' },
    '123': { unit: 'Pa', name: 'Pressure (Cloud Top)' },
    // Momentum
    '9': { unit: 'm s⁻¹', name: 'Wind Speed (Gust)' },
    '10': { unit: 'm s⁻¹', name: 'U Wind Component' },
    '11': { unit: 'm s⁻¹', name: 'V Wind Component' }
};

// Color scale function
function getColor(value, min, max) {
    const ratio = (value - min) / (max - min);
    const colors = [
        [49, 54, 149],     // Dark blue
        [69, 117, 180],    // Blue
        [116, 173, 209],   // Light blue
        [171, 217, 233],   // Very light blue
        [224, 243, 248],   // Almost white
        [255, 255, 204],   // Light yellow
        [254, 224, 144],   // Yellow
        [253, 174, 97],    // Orange
        [244, 109, 67],    // Red-orange
        [215, 48, 39],     // Red
        [165, 0, 38]       // Dark red
    ];
    
    const colorIndex = Math.floor(ratio * (colors.length - 1));
    const nextIndex = Math.min(colorIndex + 1, colors.length - 1);
    const localRatio = (ratio * (colors.length - 1)) - colorIndex;
    
    const color1 = colors[colorIndex];
    const color2 = colors[nextIndex];
    
    const r = Math.round(color1[0] + (color2[0] - color1[0]) * localRatio);
    const g = Math.round(color1[1] + (color2[1] - color1[1]) * localRatio);
    const b = Math.round(color1[2] + (color2[2] - color1[2]) * localRatio);
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Generate grid coordinates for entire Florida
function generateFloridaGrid(gridSize = 25) {
    const bounds = {
        north: 31.0,   // Northern Florida
        south: 24.5,   // Southern Florida (Keys)
        east: -80.0,   // Eastern coastline
        west: -87.0    // Western Panhandle
    };
    
    const latStep = (bounds.north - bounds.south) / Math.sqrt(gridSize);
    const lonStep = (bounds.east - bounds.west) / Math.sqrt(gridSize);
    
    const coordinates = [];
    let gridId = 0;
    
    for (let lat = bounds.south; lat <= bounds.north; lat += latStep) {
        for (let lon = bounds.west; lon <= bounds.east; lon += lonStep) {
            coordinates.push({ gridId, lat, lon });
            gridId++;
        }
    }
    
    return coordinates;
}

// Load CSV data
async function loadCSVData(category, filename) {
    try {
        showLoading(true);
        const response = await fetch(`data/${category}/${filename}.csv`);
        const text = await response.text();
        
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',');
        timeColumns = headers.slice(1); // All columns except 'Grid'
        
        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            const gridId = parseInt(values[0]);
            const timeSeriesData = values.slice(1).map(v => parseFloat(v));
            data.push({ gridId, values: timeSeriesData });
        }
        
        showLoading(false);
        return data;
    } catch (error) {
        console.error('Error loading CSV data:', error);
        showLoading(false);
        alert('Error loading data. Please check the file path and try again.');
        return null;
    }
}

// Create data visualization layer
function createDataLayer(data, timeIndex) {
    const gridCoordinates = generateFloridaGrid(data.length);
    const values = data.map(d => d.values[timeIndex]).filter(v => !isNaN(v));
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    
    const markers = [];
    
    data.forEach((gridData, index) => {
        if (index < gridCoordinates.length) {
            const coord = gridCoordinates[index];
            const value = gridData.values[timeIndex];
            
            if (!isNaN(value)) {
                const color = getColor(value, minVal, maxVal);
                const opacity = 0.7;
                
                const circle = L.circleMarker([coord.lat, coord.lon], {
                    radius: 5,
                    fillColor: color,
                    color: 'white',
                    weight: 0.5,
                    opacity: 0.4,
                    fillOpacity: 0.25
                });
                
                circle.bindPopup(`
                    <div style="text-align: center; min-width: 200px;">
                        <h4 style="margin: 0 0 8px 0; color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px;">Grid Point ${coord.gridId}</h4>
                        <div style="text-align: left;">
                            <p style="margin: 4px 0;"><strong>📊 Value:</strong> ${value.toFixed(3)} ${dataDescriptions[document.getElementById('dataFile').value]?.unit || ''}</p>
                            <p style="margin: 4px 0;"><strong>📍 Coordinates:</strong> ${coord.lat.toFixed(4)}°N, ${Math.abs(coord.lon).toFixed(4)}°W</p>
                            <p style="margin: 4px 0;"><strong>🕒 Time:</strong> ${timeColumns[timeIndex].replace(/_/g, ' ').replace('.npy', '')}</p>
                            <p style="margin: 4px 0 0 0; font-size: 11px; color: #7f8c8d;"><strong>Data Type:</strong> ${dataDescriptions[document.getElementById('dataFile').value]?.name || 'Unknown'}</p>
                        </div>
                    </div>
                `);
                
                markers.push(circle);
            }
        }
    });
    
    updateLegend(minVal, maxVal);
    return L.layerGroup(markers);
}

// Update legend
function updateLegend(min, max) {
    const legend = document.getElementById('legend');
    const legendTitle = document.getElementById('legendTitle');
    const minValue = document.getElementById('minValue');
    const maxValue = document.getElementById('maxValue');
    const unitLabel = document.getElementById('unitLabel');
    
    const selectedFile = document.getElementById('dataFile').value;
    const description = dataDescriptions[selectedFile];
    
    if (description) {
        legendTitle.textContent = description.name;
        unitLabel.textContent = `Unit: ${description.unit}`;
    }
    
    minValue.textContent = min.toFixed(2);
    maxValue.textContent = max.toFixed(2);
    legend.style.display = 'block';
}

// Show/hide loading indicator
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

// Event handlers
document.getElementById('dataCategory').addEventListener('change', function() {
    const category = this.value;
    const dataFileSelect = document.getElementById('dataFile');
    
    dataFileSelect.innerHTML = '<option value="">Select data type...</option>';
    dataFileSelect.disabled = !category;
    
    if (category && dataFiles[category]) {
        dataFiles[category].forEach(file => {
            const option = document.createElement('option');
            option.value = file.value;
            option.textContent = file.label;
            dataFileSelect.appendChild(option);
        });
        dataFileSelect.disabled = false;
    }
    
    // Reset other controls
    document.getElementById('timeSlider').disabled = true;
    document.getElementById('loadData').disabled = true;
    document.getElementById('timeLabel').textContent = 'Select data first';
});

document.getElementById('dataFile').addEventListener('change', function() {
    const hasFile = this.value !== '';
    document.getElementById('loadData').disabled = !hasFile;
});

document.getElementById('loadData').addEventListener('click', async function() {
    const category = document.getElementById('dataCategory').value;
    const filename = document.getElementById('dataFile').value;
    
    if (category && filename) {
        currentData = await loadCSVData(category, filename);
        
        if (currentData) {
            // Enable time slider
            const timeSlider = document.getElementById('timeSlider');
            timeSlider.max = timeColumns.length - 1;
            timeSlider.value = 0;
            timeSlider.disabled = false;
            
            // Update initial visualization
            updateVisualization(0);
            
            // Update time label
            updateTimeLabel(0);
        }
    }
});

document.getElementById('timeSlider').addEventListener('input', function() {
    const timeIndex = parseInt(this.value);
    updateVisualization(timeIndex);
    updateTimeLabel(timeIndex);
});

document.getElementById('clearData').addEventListener('click', function() {
    if (currentDataLayer) {
        map.removeLayer(currentDataLayer);
        currentDataLayer = null;
    }
    document.getElementById('legend').style.display = 'none';
    
    // Reset controls
    document.getElementById('dataCategory').value = '';
    document.getElementById('dataFile').innerHTML = '<option value="">Select data type...</option>';
    document.getElementById('dataFile').disabled = true;
    document.getElementById('timeSlider').disabled = true;
    document.getElementById('timeSlider').value = 0;
    document.getElementById('loadData').disabled = true;
    document.getElementById('timeLabel').textContent = 'Select data first';
    
    currentData = null;
    timeColumns = [];
});

function updateVisualization(timeIndex) {
    if (currentDataLayer) {
        map.removeLayer(currentDataLayer);
    }
    
    if (currentData && timeIndex >= 0 && timeIndex < timeColumns.length) {
        currentDataLayer = createDataLayer(currentData, timeIndex);
        currentDataLayer.addTo(map);
    }
}

function updateTimeLabel(timeIndex) {
    const timeLabel = document.getElementById('timeLabel');
    if (timeColumns.length > 0 && timeIndex >= 0 && timeIndex < timeColumns.length) {
        const timestamp = timeColumns[timeIndex];
        // Parse timestamp to more readable format
        const match = timestamp.match(/(\d{8})_(\d{2})_/);
        if (match) {
            const date = match[1];
            const hour = match[2];
            const year = date.substring(0, 4);
            const month = date.substring(4, 6);
            const day = date.substring(6, 8);
            timeLabel.textContent = `${year}-${month}-${day} ${hour}:00`;
        } else {
            timeLabel.textContent = timestamp;
        }
    }
}

// Responsive design adjustments
function adjustMapForMobile() {
    if (window.innerWidth <= 768) {
        map.zoomControl.setPosition('topright');
    }
}

adjustMapForMobile();
window.addEventListener('resize', adjustMapForMobile); 