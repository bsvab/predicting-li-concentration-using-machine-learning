// Fetch and display actual lithium data
fetch('/api/actual_li')
    .then(response => {
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        L.geoJSON(data, {
            pointToLayer: pointToLayerActual 
        }).addTo(actualLiLayer);
    })
    .catch(error => console.error('Error fetching data:', error));

// Fetch and display predicted lithium data
fetch('/api/predicted_li/SVR')
    .then(response => {
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        L.geoJSON(data, {
            pointToLayer: pointToLayerSVR  
        }).addTo(predicted_li_SVR);
    })
    .catch(error => console.error('Error fetching data:', error));


fetch('/api/predicted_li/XGB')
.then(response => {
    if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    return response.json();
})
.then(data => {
    L.geoJSON(data, {
        pointToLayer: pointToLayerXGB  
    }).addTo(predicted_li_XGB);
})
.catch(error => console.error('Error fetching data:', error));

fetch('/api/predicted_li/RF')
.then(response => {
    if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    return response.json();
})
.then(data => {
    L.geoJSON(data, {
        pointToLayer: pointToLayerRF  
    }).addTo(predicted_li_RF);
})
.catch(error => console.error('Error fetching data:', error));

fetch('/api/predicted_li/MLP')
.then(response => {
    if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    return response.json();
})
.then(data => {
    L.geoJSON(data, {
        pointToLayer: pointToLayerMLP  
    }).addTo(predicted_li_MLP);
})
.catch(error => console.error('Error fetching data:', error));

fetch('/api/predicted_li/GB')
.then(response => {
    if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    return response.json();
})
.then(data => {
    L.geoJSON(data, {
        pointToLayer: pointToLayerGB 
    }).addTo(predicted_li_GB);
})
.catch(error => console.error('Error fetching data:', error));


// Function to determine Li circle size based on concentrations
function CircleSize(Li_predicted) {
    if (Li_predicted < 80) return 2;
    else if (Li_predicted >= 80 && Li_predicted < 120) return 4;
    else if (Li_predicted >= 120 && Li_predicted < 160) return 6;
    else if (Li_predicted >= 160 && Li_predicted < 200) return 8;
    else if (Li_predicted >= 200 && Li_predicted < 240) return 10;
    else if (Li_predicted >= 240 && Li_predicted < 280) return 12;
    else if (Li_predicted >= 280 && Li_predicted < 320) return 14;
    else if (Li_predicted >= 320 && Li_predicted < 360) return 16;
    else if (Li_predicted >= 360 && Li_predicted < 400) return 18;
    else return 20;
}


// Define pointToLayer function for Actual Li
let pointToLayerActual = function (feature, latlng) {
    //console.log(feature.properties);
    let Li = feature.properties.Li;
    
    // if Li is undefined or null, skip rendering this point
    if (Li === undefined || Li === null) {
        console.error('Li property is missing:', feature);
        return null;  
    }

    let radius = CircleSize(Li);
    let LiFormatted = parseFloat(Li).toFixed(2);
    let TDS = feature.properties.TDS;
    let TDSFormatted = TDS ? parseFloat(TDS).toFixed(2) : 'N/A';
    let FORMSIMPLE = feature.properties.FORMSIMPLE;
    let popupContent = `Formation: ${FORMSIMPLE || 'N/A'}<br>` +
                        `Actual Li: ${LiFormatted} ppm<br>` +
                        `TDS: ${TDSFormatted} ppm`;

    return L.circleMarker(latlng, {
        radius: radius,
        color: 'red', 
        fillColor: 'red',
        fillOpacity: 0.45,
        weight: 0.4
    }).bindPopup(popupContent);
};

// Define pointToLayer function for SVR predicted Li
let pointToLayerSVR = function (feature, latlng) {
    //console.log(feature.properties);
    let Li_predicted = feature.properties.Li_predicted;

    // Validate and adjust Li_predicted
    if (Li_predicted !== undefined && Li_predicted !== null) {
        // Convert Li_predicted to a number, and set it to zero if it's less than zero
        Li_predicted = parseFloat(Li_predicted);
        Li_predicted = Li_predicted < 0 ? 0 : Li_predicted;
        Li_predicted = Li_predicted.toFixed(2);  // Format to two decimal places
    } else {
        console.error('Li_predicted property is missing or invalid:', feature);
        return null; // Optionally skip rendering if Li_predicted is crucial and missing
    }

    let radius = CircleSize(Li_predicted);
    let TDS = feature.properties.TDS;
    let TDSFormatted = TDS ? parseFloat(TDS).toFixed(2) : 'N/A'; 
    let FORMSIMPLE = feature.properties.FORMSIMPLE;

    let popupContent = `Formation: ${FORMSIMPLE || 'N/A'}<br>` +
                        `Li Predicted: ${Li_predicted} ppm<br>` +
                        `TDS: ${TDSFormatted} ppm`;
    return L.circleMarker(latlng, {
        radius: radius,
        color: '#3388ff', // Blue color 
        fillColor: '#3388ff',
        fillOpacity: 0.45,
        weight: 0.4
    }).bindPopup(popupContent);
};

// Define pointToLayer function for XGB predicted Li
let pointToLayerXGB = function (feature, latlng) {
    //console.log(feature.properties);
    let Li_predicted = feature.properties.Li_predicted;
    
    // Check and adjust Li_predicted value
    if (Li_predicted !== undefined && Li_predicted !== null) {
        // Convert Li_predicted to a number and ensure it is not negative
        Li_predicted = parseFloat(Li_predicted);
        Li_predicted = Li_predicted < 0 ? 0 : Li_predicted;
        Li_predicted = Li_predicted.toFixed(2); // Format to two decimal places
    } else {
        console.error('Li_predicted property is missing or invalid:', feature);
        return null; // Optionally skip rendering if Li_predicted is crucial and missing
    }

    let radius = CircleSize(Li_predicted);
    let TDS = feature.properties.TDS;
    let TDSFormatted = TDS ? parseFloat(TDS).toFixed(2) : 'N/A'; 
    let FORMSIMPLE = feature.properties.FORMSIMPLE;

    let popupContent = `Formation: ${FORMSIMPLE || 'N/A'}<br>` +
                        `Li Predicted: ${Li_predicted} ppm<br>` +
                        `TDS: ${TDSFormatted} ppm`;
    return L.circleMarker(latlng, {
        radius: radius,
        color: 'green' , 
        fillColor: 'green',
        fillOpacity: 0.35,
        weight: 0.4
    }).bindPopup(popupContent);
};


// Define pointToLayer function for RF predicted Li
let pointToLayerRF = function (feature, latlng) {
    //console.log(feature.properties);
    let Li_predicted = feature.properties.Li_predicted;
    
    // Check and adjust Li_predicted value
    if (Li_predicted !== undefined && Li_predicted !== null) {
        // Convert Li_predicted to a number and ensure it is not negative
        Li_predicted = parseFloat(Li_predicted);
        Li_predicted = Li_predicted < 0 ? 0 : Li_predicted;
        Li_predicted = Li_predicted.toFixed(2); // Format to two decimal places
    } else {
        console.error('Li_predicted property is missing or invalid:', feature);
        return null; 
    }

    let radius = CircleSize(Li_predicted);
    let TDS = feature.properties.TDS;
    let TDSFormatted = TDS ? parseFloat(TDS).toFixed(2) : 'N/A'; 
    let FORMSIMPLE = feature.properties.FORMSIMPLE;

    let popupContent = `Formation: ${FORMSIMPLE || 'N/A'}<br>` +
                        `Li Predicted: ${Li_predicted} ppm<br>` +
                        `TDS: ${TDSFormatted} ppm`;
    return L.circleMarker(latlng, {
        radius: radius,
        color: 'gray' , 
        fillColor: 'gray',
        fillOpacity: 0.35,
        weight: 0.4
    }).bindPopup(popupContent);
};

// Define pointToLayer function for MLP predicted Li
let pointToLayerMLP = function (feature, latlng) {
    //console.log(feature.properties);
    let Li_predicted = feature.properties.Li_predicted;
    
    // Check and adjust Li_predicted value
    if (Li_predicted !== undefined && Li_predicted !== null) {
        // Convert Li_predicted to a number and ensure it is not negative
        Li_predicted = parseFloat(Li_predicted);
        Li_predicted = Li_predicted < 0 ? 0 : Li_predicted;
        Li_predicted = Li_predicted.toFixed(2); // Format to two decimal places
    } else {
        console.error('Li_predicted property is missing or invalid:', feature);
        return null; 
    }

    let radius = CircleSize(Li_predicted);
    let TDS = feature.properties.TDS;
    let TDSFormatted = TDS ? parseFloat(TDS).toFixed(2) : 'N/A'; 
    let FORMSIMPLE = feature.properties.FORMSIMPLE;

    let popupContent = `Formation: ${FORMSIMPLE || 'N/A'}<br>` +
                        `Li Predicted: ${Li_predicted} ppm<br>` +
                        `TDS: ${TDSFormatted} ppm`;
    return L.circleMarker(latlng, {
        radius: radius,
        color: 'orange' , 
        fillColor: 'orange',
        fillOpacity: 0.35,
        weight: 0.4
    }).bindPopup(popupContent);
};

// Define pointToLayer function for GB predicted Li
let pointToLayerGB = function (feature, latlng) {
    //console.log(feature.properties);
    let Li_predicted = feature.properties.Li_predicted;
    
    // Check and adjust Li_predicted value
    if (Li_predicted !== undefined && Li_predicted !== null) {
        // Convert Li_predicted to a number and ensure it is not negative
        Li_predicted = parseFloat(Li_predicted);
        Li_predicted = Li_predicted < 0 ? 0 : Li_predicted;
        Li_predicted = Li_predicted.toFixed(2); // Format to two decimal places
    } else {
        console.error('Li_predicted property is missing or invalid:', feature);
        return null; 
    }

    let radius = CircleSize(Li_predicted);
    let TDS = feature.properties.TDS;
    let TDSFormatted = TDS ? parseFloat(TDS).toFixed(2) : 'N/A'; 
    let FORMSIMPLE = feature.properties.FORMSIMPLE;

    let popupContent = `Formation: ${FORMSIMPLE || 'N/A'}<br>` +
                        `Li Predicted: ${Li_predicted} ppm<br>` +
                        `TDS: ${TDSFormatted} ppm`;
    return L.circleMarker(latlng, {
        radius: radius,
        color: 'purple' , 
        fillColor: 'purple',
        fillOpacity: 0.35,
        weight: 0.4
    }).bindPopup(popupContent);
};