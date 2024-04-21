// Initialize the map
let map5 = L.map('map5').setView([29.7633, -95.3633], 6); 

//Define OpenStreetMap tiles
let osmTiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
});

// Define Topographic map tiles 
let topoTiles = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    attribution: 'Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)'
});

// Add OpenStreetMap tiles to the map 
osmTiles.addTo(map5);

let actualLiLayer = L.layerGroup().addTo(map5);
let predicted_li_XGB = L.layerGroup().addTo(map5);
let predicted_li_SVR = L.layerGroup();
let predicted_li_RF = L.layerGroup();
let predicted_li_MLP = L.layerGroup();
let predicted_li_GB = L.layerGroup();

// Define all basemaps 
let baseMaps = {
    "OpenStreetMap": osmTiles,
    "Topographic": topoTiles
};


let overlayMaps = {
    "Actual Li": actualLiLayer,
    "Predicted Li by SVR": predicted_li_SVR,
    "Predicted Li by XGB": predicted_li_XGB,
    "Predited Li by RF":predicted_li_RF,
    "Predited Li by MLP":predicted_li_MLP,
    "Predited Li by GB":predicted_li_GB

};
// Add layer control to the map
let layerControl = L.control.layers(baseMaps, overlayMaps, {collapsed: false}).addTo(map5);

