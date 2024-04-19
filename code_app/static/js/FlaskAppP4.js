const express = require('express');
const path = require('path');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 5000;

// Configure multer for handling file uploads
const upload = multer({ dest: 'uploads/' });

// Serve static files from the 'code_app' directory
app.use(express.static(path.join(__dirname, 'code_app')));

// Route for serving the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'code_app', 'templates', 'MachineToolIndex.html'));
});


// Route for serving other static files if needed
app.get('/:filename', (req, res) => {
    const filename = req.params.filename;
    res.sendFile(path.join(__dirname, 'code_app', 'static', filename));
});

// Route for running the model
app.post('/run_model', upload.single('file'), (req, res) => {
    // Get the uploaded file and selected model from request
    const file = req.file;
    const selectedModel = req.body.model;

    // Implement your model logic here
    // This is just a placeholder
    const results = {
        'R-squared': 0.75,
        'Accuracy': 0.85
    };

    // Return the results as JSON
    res.json(results);
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});