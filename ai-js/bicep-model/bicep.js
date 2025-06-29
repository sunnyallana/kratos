const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

const BICEP_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
];

// Generate dataset columns
const DATASET_COLUMNS = ["label"];
for (const landmark of BICEP_LANDMARKS) {
    const lowerLandmark = landmark.toLowerCase();
    DATASET_COLUMNS.push(
        `${lowerLandmark}_x`,
        `${lowerLandmark}_y`,
        `${lowerLandmark}_z`,
        `${lowerLandmark}_v`
    );
}

// Set random seed for reproducibility
const SEED = 42;
tf.initializers.glorotNormal({seed: SEED});
tf.initializers.zeros({seed: SEED});

// Helper functions
function createRequiredDirectories() {
    const requiredDirs = ["./model", "./logs"];
    for (const dir of requiredDirs) {
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
    }
}

async function loadDataset(filePath) {
    return new Promise((resolve, reject) => {
        const results = [];
        fs.createReadStream(filePath)
            .pipe(csv())
            .on('data', (data) => results.push(data))
            .on('end', () => {
                resolve(results);
            })
            .on('error', (error) => {
                reject(error);
            });
    });
}

function analyzeDataset(dataset) {
    console.log("Dataset columns:", Object.keys(dataset[0]));
    console.log(`Dataset shape: ${dataset.length} rows x ${Object.keys(dataset[0]).length} columns\n`);
    
    const labelCounts = {};
    dataset.forEach(row => {
        const label = row.label;
        labelCounts[label] = (labelCounts[label] || 0) + 1;
    });
    console.log("Class distribution:", labelCounts, "\n");
    
    // Check for missing values
    let hasMissing = false;
    for (const row of dataset) {
        for (const val of Object.values(row)) {
            if (val === null || val === undefined || val === '') {
                hasMissing = true;
                break;
            }
        }
        if (hasMissing) break;
    }
    console.log(`Missing values present: ${hasMissing}\n`);
    
    return dataset;
}

function cleanDuplicateData(dataset, outputFilename = "cleaned_train.csv") {
    const uniqueRows = [];
    const seen = new Set();
    
    for (const row of dataset) {
        const rowStr = JSON.stringify(row);
        if (!seen.has(rowStr)) {
            seen.add(rowStr);
            uniqueRows.push(row);
        }
    }
    
    const originalSize = dataset.length;
    const cleanedSize = uniqueRows.length;
    
    // Convert to CSV and save
    const header = Object.keys(uniqueRows[0]).join(',');
    const csvContent = [header, ...uniqueRows.map(row => 
        Object.values(row).join(','))].join('\n');
    
    fs.writeFileSync(outputFilename, csvContent);
    console.log(`Removed ${originalSize - cleanedSize} duplicate rows`);
    console.log(`Cleaned dataset saved to ${outputFilename}`);
    
    return uniqueRows;
}

function preprocessData(dataset) {
    // Encode labels: 'C' (Correct) = 0, 'L' (Low) = 1
    const processed = dataset.map(row => {
        const newRow = {...row};
        if (newRow.label === 'C') newRow.label = '0';
        if (newRow.label === 'L') newRow.label = '1';
        return newRow;
    });
    
    return processed;
}

class StandardScaler {
    constructor() {
        this.mean = null;
        this.std = null;
    }
    
    async fit(data) {
        const numericData = data.map(row => 
            Object.values(row)
                .filter((_, i) => i !== 0) // Exclude label column
                .map(val => parseFloat(val))
        );
        
        const tensor = tf.tensor2d(numericData);
        this.mean = tensor.mean(0);
        
        // Calculate standard deviation manually
        const squaredDiffs = tensor.sub(this.mean).square();
        const variance = squaredDiffs.mean(0);
        this.std = variance.sqrt();
        
        // Dispose temporary tensors to free memory
        squaredDiffs.dispose();
        variance.dispose();
    }
    
    transform(data) {
        if (!this.mean || !this.std) {
            throw new Error("Scaler has not been fitted yet");
        }
        
        const numericData = data.map(row => 
            Object.values(row)
                .filter((_, i) => i !== 0) // Exclude label column
                .map(val => parseFloat(val))
        );
        
        const tensor = tf.tensor2d(numericData);
        const scaled = tensor.sub(this.mean).div(this.std);
        const result = scaled.arraySync();
        
        // Dispose tensors to free memory
        tensor.dispose();
        scaled.dispose();
        
        return result;
    }
    
    save(filePath) {
        const data = {
            mean: this.mean.arraySync(),
            std: this.std.arraySync()
        };
        fs.writeFileSync(filePath, JSON.stringify(data));
    }
    
    static load(filePath) {
        const data = JSON.parse(fs.readFileSync(filePath));
        const scaler = new StandardScaler();
        scaler.mean = tf.tensor(data.mean);
        scaler.std = tf.tensor(data.std);
        return scaler;
    }
}


function toCategorical(labels, numClasses) {
    const categorical = labels.map(label => {
        const arr = new Array(numClasses).fill(0);
        arr[parseInt(label)] = 1;
        return arr;
    });
    return tf.tensor2d(categorical);
}

function displayModelArchitecture(model) {
    console.log("Neural Network Architecture:");
    console.log("-".repeat(40));
    model.layers.forEach((layer, index) => {
        const config = layer.getConfig();
        console.log(`Layer ${index + 1}: ${config.units || 'N/A'} units, activation: ${config.activation || 'None'}`);
    });
}

function evaluateModelPerformance(model, testFeatures, testLabelsCategorical) {
    // Make predictions
    const predictions = model.predict(testFeatures);
    const predictedClasses = predictions.argMax(-1).arraySync();
    const actualClasses = testLabelsCategorical.argMax(-1).arraySync();
    
    // Calculate confusion matrix
    let confusionMatrix = [[0, 0], [0, 0]];
    for (let i = 0; i < actualClasses.length; i++) {
        const actual = actualClasses[i];
        const predicted = predictedClasses[i];
        confusionMatrix[actual][predicted]++;
    }
    
    // Calculate precision, recall, f1
    const metrics = {
        precision: [],
        recall: [],
        f1_score: []
    };
    
    for (let cls = 0; cls < 2; cls++) {
        const tp = confusionMatrix[cls][cls];
        const fp = confusionMatrix[1-cls][cls];
        const fn = confusionMatrix[cls][1-cls];
        
        const precision = tp / (tp + fp);
        const recall = tp / (tp + fn);
        const f1 = 2 * (precision * recall) / (precision + recall);
        
        metrics.precision.push(roundMetricScore(precision));
        metrics.recall.push(roundMetricScore(recall));
        metrics.f1_score.push(roundMetricScore(f1));
    }
    
    return {
        confusion_matrix: confusionMatrix,
        ...metrics
    };
}

function roundMetricScore(score) {
    return Math.round(score * 1000) / 1000;
}

async function main() {
    console.log("=== Bicep Exercise Classification Training ===\n");
    
    // Create necessary directories
    createRequiredDirectories();
    
    // Load and analyze training dataset
    console.log("Loading and analyzing training dataset...");
    let dataset = await loadDataset("./train.csv");
    dataset = analyzeDataset(dataset);
    
    if (!dataset || dataset.length === 0) {
        console.log("Cannot proceed without training data. Please ensure train.csv exists.");
        return;
    }
    
    // Clean duplicates
    const cleanedDataset = cleanDuplicateData(dataset);
    
    // Preprocess data (encode labels)
    const processedDataset = preprocessData(cleanedDataset);
    
    // Handle feature scaling
    let featureScaler;
    const scalerFilePath = "./model/input_scaler.json";
    
    if (fs.existsSync(scalerFilePath)) {
        console.log("Loading existing feature scaler");
        featureScaler = StandardScaler.load(scalerFilePath);
    } else {
        console.log("Creating new StandardScaler for feature normalization...");
        featureScaler = new StandardScaler();
        await featureScaler.fit(processedDataset);
        featureScaler.save(scalerFilePath);
        console.log("Created and saved new feature scaler");
    }

    // Prepare features and labels
    const features = featureScaler.transform(processedDataset);
    const labels = processedDataset.map(row => row.label);
    const labelsCategorical = toCategorical(labels, 2);
    
    // Convert to tensors
    const featuresTensor = tf.tensor2d(features);
    const labelsTensor = labelsCategorical;
    
    // Split data for training and validation
    const splitIdx = Math.floor(featuresTensor.shape[0] * 0.8);
    const trainFeatures = featuresTensor.slice(0, splitIdx);
    const valFeatures = featuresTensor.slice(splitIdx);
    const trainLabels = labelsTensor.slice(0, splitIdx);
    const valLabels = labelsTensor.slice(splitIdx);
    
    console.log(`Training set shape: ${trainFeatures.shape}`);
    console.log(`Validation set shape: ${valFeatures.shape}`);
    
    // Build the model
    console.log("\nBuilding the model...");
    const model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.dense({
        units: 36,
        inputShape: [36],
        activation: 'relu'
    }));
    
    // Hidden layers
    model.add(tf.layers.dense({units: 256, activation: 'relu'}));
    model.add(tf.layers.dense({units: 256, activation: 'relu'}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    
    // Output layer
    model.add(tf.layers.dense({
        units: 2,
        activation: 'softmax'
    }));
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    displayModelArchitecture(model);
    
    // Train the model
    console.log("\nTraining the model...");
    const history = await model.fit(trainFeatures, trainLabels, {
        epochs: 50,  // Reduced from 100 for faster training
        batchSize: 16,
        validationData: [valFeatures, valLabels],
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}, val_acc = ${logs.val_acc.toFixed(4)}`);
            }
        }
    });
    
    // Evaluate on validation set
    console.log("\n" + "=".repeat(60));
    console.log("VALIDATION SET EVALUATION");
    console.log("=".repeat(60));
    
    const valMetrics = evaluateModelPerformance(model, valFeatures, valLabels);
    console.log(`Precision scores: ${valMetrics.precision}`);
    console.log(`Recall scores: ${valMetrics.recall}`);
    console.log(`F1 scores: ${valMetrics.f1_score}`);
    console.log(`Confusion Matrix:\n${valMetrics.confusion_matrix.map(row => row.join(' ')).join('\n')}`);
    
    // Test set evaluation (if available)
    if (fs.existsSync("./test.csv")) {
        console.log("\n" + "=".repeat(60));
        console.log("TEST SET EVALUATION");
        console.log("=".repeat(60));
        
        let testDataset = await loadDataset("./test.csv");
        testDataset = preprocessData(testDataset);
        
        if (testDataset && testDataset.length > 0) {
            const testFeatures = featureScaler.transform(testDataset);
            const testLabels = testDataset.map(row => row.label);
            const testLabelsCategorical = toCategorical(testLabels, 2);
            
            const testFeaturesTensor = tf.tensor2d(testFeatures);
            const testLabelsTensor = testLabelsCategorical;
            
            const testMetrics = evaluateModelPerformance(model, testFeaturesTensor, testLabelsTensor);
            console.log(`Test Precision scores: ${testMetrics.precision}`);
            console.log(`Test Recall scores: ${testMetrics.recall}`);
            console.log(`Test F1 scores: ${testMetrics.f1_score}`);
            console.log(`Test Confusion Matrix:\n${testMetrics.confusion_matrix.map(row => row.join(' ')).join('\n')}`);
        }
    }
    
    // Save the model
    console.log("\nSaving the trained model...");
    const savePath = 'file://./model/bicep_classifier';
    await model.save(savePath);
    console.log(`Model saved successfully to ${savePath}`);
    console.log("\n=== Training Pipeline Completed Successfully! ===");
}

main().catch(console.error);