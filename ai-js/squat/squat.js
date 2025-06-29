const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { execSync } = require('child_process');

class SquatPostureMLPipeline {
  constructor(randomState = 1234) {
    this.randomState = randomState;
    this.model = null;
    this.scaler = null;
    this.isTrained = false;
    this.featureNames = null;
    this.labelMapping = { 'down': 0, 'up': 1 };

    // Important landmarks for squat detection
    this.importantLandmarks = [
      "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP",
      "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ];

    // Generate column names
    this.columns = this.generateColumnNames();
  }

  generateColumnNames() {
    const columns = ["label"];
    for (const lm of this.importantLandmarks) {
      const lowerLm = lm.toLowerCase();
      columns.push(
        `${lowerLm}_x`,
        `${lowerLm}_y`,
        `${lowerLm}_z`,
        `${lowerLm}_v`
      );
    }
    return columns;
  }

  loadData(filePath) {
    return new Promise((resolve, reject) => {
      const results = [];
      fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => {
          console.log(`Data loaded successfully from ${filePath}`);
          console.log(`Shape: ${results.length} rows x ${Object.keys(results[0]).length} columns`);
          resolve(results);
        })
        .on('error', (error) => {
          console.error(`Error loading data: ${error}`);
          reject(error);
        });
    });
  }

  preprocessData(data) {
    // Create a copy to avoid modifying original data
    const processedData = JSON.parse(JSON.stringify(data));

    // Convert string labels to numeric
    processedData.forEach(row => {
      row.label = this.labelMapping[row.label.toLowerCase()];
    });

    // Separate features and labels
    const features = processedData.map(row => {
      const featureRow = {};
      for (const key in row) {
        if (key !== 'label') {
          featureRow[key] = parseFloat(row[key]);
        }
      }
      return featureRow;
    });

    const labels = processedData.map(row => row.label);

    // Store feature names
    this.featureNames = Object.keys(features[0]);

    console.log(`Features shape: ${features.length} x ${this.featureNames.length}`);
    
    const labelCounts = labels.reduce((acc, label) => {
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {});
    
    console.log(`Labels distribution: ${JSON.stringify(labelCounts)}`);

    return { features, labels };
  }

  async splitData(features, labels, testSize = 0.2) {
    // Convert to tensors for splitting
    const featuresTensor = tf.tensor2d(features.map(row => 
      Object.values(row).map(val => parseFloat(val)))
    );
    const labelsTensor = tf.tensor1d(labels);

    // Calculate split index
    const splitIdx = Math.floor(featuresTensor.shape[0] * (1 - testSize));

    // Split the data
    const trainFeatures = featuresTensor.slice(0, splitIdx);
    const testFeatures = featuresTensor.slice(splitIdx);
    const trainLabels = labelsTensor.slice(0, splitIdx);
    const testLabels = labelsTensor.slice(splitIdx);

    console.log(`Training set size: ${trainFeatures.shape[0]}`);
    console.log(`Test set size: ${testFeatures.shape[0]}`);

    return {
      trainFeatures,
      testFeatures,
      trainLabels,
      testLabels
    };
  }

  async scaleFeatures(trainFeatures, testFeatures) {
    this.scaler = new StandardScaler();
    await this.scaler.fit(trainFeatures.arraySync().map(row => 
      Object.values(row).map(val => parseFloat(val)))
    );

    const trainFeaturesScaled = this.scaler.transform(trainFeatures.arraySync());
    const testFeaturesScaled = this.scaler.transform(testFeatures.arraySync());

    console.log("Features scaled successfully");

    return {
      trainFeaturesScaled: tf.tensor2d(trainFeaturesScaled),
      testFeaturesScaled: tf.tensor2d(testFeaturesScaled)
    };
  }

  async trainModel(trainFeatures, trainLabels, modelParams = {}) {
    // Default parameters
    const defaultParams = {
      learningRate: 0.001,
      epochs: 100,
      batchSize: 32,
      verbose: 1
    };
    
    const params = { ...defaultParams, ...modelParams };

    // Build the model
    this.model = tf.sequential();
    
    // Input layer
    this.model.add(tf.layers.dense({
      units: 64,
      inputShape: [trainFeatures.shape[1]],
      activation: 'relu'
    }));
    
    // Hidden layers
    this.model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    
    // Output layer
    this.model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));

    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(params.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    // Train the model
    await this.model.fit(trainFeatures, trainLabels, {
      epochs: params.epochs,
      batchSize: params.batchSize,
      verbose: params.verbose
    });

    this.isTrained = true;
    console.log("Model trained successfully");

    return this.model;
  }

  async evaluateModel(testFeatures, testLabels) {
    if (!this.isTrained) {
      throw new Error("Model must be trained before evaluation");
    }

    // Make predictions
    const predictions = this.model.predict(testFeatures);
    const predictedClasses = predictions.round().arraySync();
    const predictedProbs = predictions.arraySync();
    const actualClasses = testLabels.arraySync();

    // Calculate metrics
    const accuracy = this.calculateAccuracy(actualClasses, predictedClasses);
    const precision = this.calculatePrecision(actualClasses, predictedClasses);
    const recall = this.calculateRecall(actualClasses, predictedClasses);
    const f1 = this.calculateF1Score(actualClasses, predictedClasses);
    const cm = this.calculateConfusionMatrix(actualClasses, predictedClasses);

    // ROC curve
    const rocData = this.calculateROC(actualClasses, predictedProbs);
    const optimalThreshold = this.findOptimalThreshold(rocData.fpr, rocData.tpr, rocData.thresholds);

    const results = {
      accuracy,
      precision,
      recall,
      f1_score: f1,
      confusion_matrix: cm,
      roc_auc: rocData.auc,
      optimal_threshold: optimalThreshold,
      fpr: rocData.fpr,
      tpr: rocData.tpr,
      thresholds: rocData.thresholds,
      predictions: predictedClasses,
      prediction_probabilities: predictedProbs
    };

    return results;
  }

  calculateAccuracy(actual, predicted) {
    let correct = 0;
    for (let i = 0; i < actual.length; i++) {
      if (actual[i] === predicted[i][0]) correct++;
    }
    return correct / actual.length;
  }

  calculatePrecision(actual, predicted) {
    let tp = 0, fp = 0;
    for (let i = 0; i < actual.length; i++) {
      if (predicted[i][0] === 1) {
        if (actual[i] === 1) tp++;
        else fp++;
      }
    }
    return tp / (tp + fp) || 0;
  }

  calculateRecall(actual, predicted) {
    let tp = 0, fn = 0;
    for (let i = 0; i < actual.length; i++) {
      if (actual[i] === 1) {
        if (predicted[i][0] === 1) tp++;
        else fn++;
      }
    }
    return tp / (tp + fn) || 0;
  }

  calculateF1Score(actual, predicted) {
    const precision = this.calculatePrecision(actual, predicted);
    const recall = this.calculateRecall(actual, predicted);
    return 2 * (precision * recall) / (precision + recall) || 0;
  }

  calculateConfusionMatrix(actual, predicted) {
    const cm = [[0, 0], [0, 0]]; // [[TN, FP], [FN, TP]]
    for (let i = 0; i < actual.length; i++) {
      const a = actual[i];
      const p = predicted[i][0];
      cm[a][p]++;
    }
    return cm;
  }

  calculateROC(actual, probs) {
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const fpr = [];
    const tpr = [];
    
    thresholds.forEach(threshold => {
      let tp = 0, fp = 0, tn = 0, fn = 0;
      
      for (let i = 0; i < actual.length; i++) {
        const prediction = probs[i][0] >= threshold ? 1 : 0;
        if (actual[i] === 1) {
          if (prediction === 1) tp++;
          else fn++;
        } else {
          if (prediction === 1) fp++;
          else tn++;
        }
      }
      
      fpr.push(fp / (fp + tn) || 0);
      tpr.push(tp / (tp + fn) || 0);
    });
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < fpr.length; i++) {
      auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
    }
    
    return { fpr, tpr, thresholds, auc };
  }

  findOptimalThreshold(fpr, tpr, thresholds) {
    let optimalIdx = 0;
    let maxDiff = -1;
    
    for (let i = 0; i < tpr.length; i++) {
      const diff = tpr[i] - fpr[i];
      if (diff > maxDiff) {
        maxDiff = diff;
        optimalIdx = i;
      }
    }
    
    return thresholds[optimalIdx];
  }

  printEvaluationSummary(results) {
    console.log("\n" + "=".repeat(50));
    console.log("LOGISTIC REGRESSION MODEL EVALUATION");
    console.log("=".repeat(50));

    console.log(`Accuracy: ${results.accuracy.toFixed(4)}`);
    console.log(`ROC AUC: ${results.roc_auc.toFixed(4)}`);
    console.log(`Optimal Threshold: ${results.optimal_threshold.toFixed(4)}`);

    console.log("\nClass-wise Metrics:");
    console.log(`${'Class'.padEnd(10)} ${'Precision'.padEnd(10)} ${'Recall'.padEnd(10)} ${'F1-Score'.padEnd(10)}`);
    console.log("-".repeat(40));
    console.log(`Down`.padEnd(10) + `${results.precision.toFixed(4)}`.padEnd(10) + 
                `${results.recall.toFixed(4)}`.padEnd(10) + `${results.f1_score.toFixed(4)}`.padEnd(10));
    console.log(`Up`.padEnd(10) + `${results.precision.toFixed(4)}`.padEnd(10) + 
                `${results.recall.toFixed(4)}`.padEnd(10) + `${results.f1_score.toFixed(4)}`.padEnd(10));

    console.log(`\nConfusion Matrix:`);
    console.log(`${'Predicted'.padEnd(12)} ${'Down'.padEnd(8)} ${'Up'.padEnd(8)}`);
    console.log("-".repeat(28));
    console.log(`${'Actual Down'.padEnd(12)} ${results.confusion_matrix[0][0].toString().padEnd(8)} ${results.confusion_matrix[0][1].toString().padEnd(8)}`);
    console.log(`${'Actual Up'.padEnd(12)} ${results.confusion_matrix[1][0].toString().padEnd(8)} ${results.confusion_matrix[1][1].toString().padEnd(8)}`);
  }

  async predict(features, threshold = null) {
    if (!this.isTrained) {
      throw new Error("Model must be trained before making predictions");
    }

    // Scale features
    const scaledFeatures = this.scaler.transform(features);

    // Get probabilities
    const probabilities = this.model.predict(tf.tensor2d(scaledFeatures)).arraySync();

    // Make predictions
    let predictions;
    if (threshold === null) {
      predictions = probabilities.map(prob => prob[0] >= 0.5 ? 1 : 0);
    } else {
      predictions = probabilities.map(prob => prob[0] >= threshold ? 1 : 0);
    }

    return { predictions, probabilities };
  }

  async saveModel(filePath) {
    if (!this.isTrained) {
      throw new Error("Model must be trained before saving");
    }

    // Create model directory if it doesn't exist
    if (!fs.existsSync(path.dirname(filePath))) {
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
    }

    // Save the model
    await this.model.save(`file://${filePath}`);

    // Save the scaler and metadata
    const modelData = {
      scaler: {
        mean: this.scaler.mean.arraySync(),
        std: this.scaler.std.arraySync()
      },
      featureNames: this.featureNames,
      labelMapping: this.labelMapping,
      importantLandmarks: this.importantLandmarks
    };

    fs.writeFileSync(`${filePath}_metadata.json`, JSON.stringify(modelData));

    console.log(`Model saved to ${filePath}`);
  }

  async loadModel(filePath) {
    try {
      // Load the model
      this.model = await tf.loadLayersModel(`file://${filePath}/model.json`);

      // Load the metadata
      const modelData = JSON.parse(fs.readFileSync(`${filePath}_metadata.json`));

      // Restore the scaler
      this.scaler = new StandardScaler();
      this.scaler.mean = tf.tensor(modelData.scaler.mean);
      this.scaler.std = tf.tensor(modelData.scaler.std);

      // Restore other properties
      this.featureNames = modelData.featureNames;
      this.labelMapping = modelData.labelMapping;
      this.importantLandmarks = modelData.importantLandmarks;
      this.isTrained = true;

      console.log(`Model loaded from ${filePath}`);
    } catch (error) {
      console.error(`Error loading model: ${error}`);
      throw error;
    }
  }

  async runCompletePipeline(trainFilePath, testFilePath = null, testSize = 0.2, modelParams = {}, saveModelPath = null) {
    console.log("Starting Complete ML Pipeline for Squat Posture Detection");
    console.log("=".repeat(60));

    try {
      // 1. Load data
      console.log("\n1. Loading Data...");
      const trainData = await this.loadData(trainFilePath);

      // 2. Preprocess data
      console.log("\n2. Preprocessing Data...");
      const { features, labels } = this.preprocessData(trainData);

      // 3. Split data or load separate test set
      let trainFeatures, testFeatures, trainLabels, testLabels;
      
      if (testFilePath) {
        console.log("\n3. Loading separate test data...");
        const testData = await this.loadData(testFilePath);
        const testProcessed = this.preprocessData(testData);
        testFeatures = tf.tensor2d(testProcessed.features.map(row => 
          Object.values(row).map(val => parseFloat(val)))
        );
        testLabels = tf.tensor1d(testProcessed.labels);
        
        trainFeatures = tf.tensor2d(features.map(row => 
          Object.values(row).map(val => parseFloat(val)))
        );
        trainLabels = tf.tensor1d(labels);
      } else {
        console.log(`\n3. Splitting data (test_size=${testSize})...`);
        const splitData = await this.splitData(features, labels, testSize);
        trainFeatures = splitData.trainFeatures;
        testFeatures = splitData.testFeatures;
        trainLabels = splitData.trainLabels;
        testLabels = splitData.testLabels;
      }

      // 4. Scale features
      console.log("\n4. Scaling Features...");
      const scaledData = await this.scaleFeatures(trainFeatures, testFeatures);
      trainFeatures = scaledData.trainFeaturesScaled;
      testFeatures = scaledData.testFeaturesScaled;

      // 5. Train model
      console.log("\n5. Training Model...");
      await this.trainModel(trainFeatures, trainLabels, modelParams);

      // 6. Evaluate model
      console.log("\n6. Evaluating Model...");
      const results = await this.evaluateModel(testFeatures, testLabels);
      this.printEvaluationSummary(results);

      // 7. Save model if requested
      if (saveModelPath) {
        console.log(`\n7. Saving Model to ${saveModelPath}...`);
        await this.saveModel(saveModelPath);
      }

      console.log("\nPipeline completed successfully!");
      return results;
    } catch (error) {
      console.error("\nPipeline failed:", error);
      throw error;
    }
  }
}

class StandardScaler {
  constructor() {
    this.mean = null;
    this.std = null;
  }

  async fit(data) {
    const tensor = tf.tensor2d(data);
    this.mean = tensor.mean(0);
    
    // Calculate standard deviation
    const squaredDiffs = tensor.sub(this.mean).square();
    const variance = squaredDiffs.mean(0);
    this.std = variance.sqrt();
    
    // Clean up
    squaredDiffs.dispose();
    variance.dispose();
    tensor.dispose();
  }

  transform(data) {
    if (!this.mean || !this.std) {
      throw new Error("Scaler has not been fitted yet");
    }

    const tensor = tf.tensor2d(data);
    const scaled = tensor.sub(this.mean).div(this.std);
    const result = scaled.arraySync();
    
    // Clean up
    tensor.dispose();
    scaled.dispose();
    
    return result;
  }
}

// Example usage
(async () => {
  try {
    // Initialize pipeline
    const pipeline = new SquatPostureMLPipeline(1234);

    // Run complete pipeline
    const results = await pipeline.runCompletePipeline(
      "./train.csv",
      "./test.csv",
      0.2,
      { learningRate: 0.001, epochs: 50, batchSize: 32 },
      "./saved_model/squat_model"
    );

  } catch (error) {
    console.error("Error in pipeline:", error);
  }
})();