// Initialize the map
let map5 = L.map('map5').setView([31.995845, -103.817762], 7.5);   // CHANGE TO WHATEVER IT NEEDS TO BE FOR GULF COAST BASIN

// Add OpenStreetMap tiles to the map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map5);
