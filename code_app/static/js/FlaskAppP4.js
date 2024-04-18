const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Route for serving the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'code_app', 'templates', 'MachineToolIndex.html'));
});

// Route for running the model (assuming you'll implement this later)
app.post('/run_model', (req, res) => {
    res.json({
        'R-squared': 0.75,
        'Accuracy': 0.85
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});