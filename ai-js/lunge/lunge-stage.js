const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

class LungeStageMLPipeline {
  constructor(randomState = 42) {
    this.randomState = randomState;
    this.model = null;
    this.scaler = null;
    this.isTrained = false;
    this.featureNames = null;
    this.labelMapping = { 'I': 0, 'M': 1, 'D': 2 };
    this.classNames = ['I', 'M', 'D'];
    this.bestValLoss = Infinity;
    this.bestWeights = null;
  }

  async loadData(filePath) {
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
    const processedData = JSON.parse(JSON.stringify(data));

    // Convert string labels to numeric
    processedData.forEach(row => {
      row.label = this.labelMapping[row.label];
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
      acc[this.classNames[label]] = (acc[this.classNames[label]] || 0) + 1;
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
    // FIX: Convert labels to float32 instead of int32
    const labelsTensor = tf.tensor1d(labels, 'float32');

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
    // Calculate mean and std for scaling
    const { mean, variance } = tf.moments(trainFeatures, 0);
    const std = variance.sqrt();
    
    // Handle division by zero (replace with 1)
    const safeStd = tf.where(
      tf.equal(std, 0),
      tf.onesLike(std),
      std
    );
    
    this.scaler = { mean, std: safeStd };

    const trainFeaturesScaled = trainFeatures.sub(mean).div(safeStd);
    const testFeaturesScaled = testFeatures.sub(mean).div(safeStd);

    console.log("Features scaled successfully");

    return {
      trainFeaturesScaled,
      testFeaturesScaled
    };
  }

  async trainModel(trainFeatures, trainLabels, modelParams = {}) {
    const defaultParams = {
      learningRate: 0.001,
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      verbose: 1,
      classWeight: { 0: 1, 1: 1.5, 2: 1.2 }
    };
    
    const params = { ...defaultParams, ...modelParams };

    // Initialize best loss tracking
    this.bestValLoss = Infinity;
    this.bestWeights = null;

    // Build the model
    this.model = tf.sequential();
    
    // Input layer
    this.model.add(tf.layers.dense({
      units: 128,
      inputShape: [trainFeatures.shape[1]],
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));
    
    // Hidden layers with dropout for regularization
    this.model.add(tf.layers.dropout({ rate: 0.3 }));
    this.model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));
    
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({
      units: 32,
      activation: 'relu'
    }));
    
    // Output layer
    this.model.add(tf.layers.dense({
      units: 3,
      activation: 'softmax'
    }));

    // Compile with custom metrics to track precision and recall
    this.model.compile({
      optimizer: tf.train.adam(params.learningRate),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });

    // Custom early stopping implementation
    const history = await this.model.fit(trainFeatures, trainLabels, {
      epochs: params.epochs,
      batchSize: params.batchSize,
      validationSplit: params.validationSplit,
      verbose: params.verbose,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          // Track best weights based on validation loss
          if (logs.val_loss < this.bestValLoss) {
            this.bestValLoss = logs.val_loss;
            this.bestWeights = await this.model.getWeights();
            console.log(`New best validation loss: ${this.bestValLoss.toFixed(4)}`);
          }
          
          // Early stopping if no improvement for 20 epochs
          if (epoch > 20 && logs.val_loss > this.bestValLoss * 1.01) {
            console.log(`Early stopping at epoch ${epoch}`);
            this.model.stopTraining = true;
          }
        },
        onTrainEnd: async () => {
          // Restore best weights
          if (this.bestWeights) {
            console.log('Restoring best weights...');
            await this.model.setWeights(this.bestWeights);
          }
        }
      }
    });

    this.isTrained = true;
    console.log("Model trained successfully");

    return history;
  }

  async evaluateModel(testFeatures, testLabels) {
    if (!this.isTrained) {
      throw new Error("Model must be trained before evaluation");
    }

    // Make predictions
    const predictions = this.model.predict(testFeatures);
    const predictedClasses = predictions.argMax(1).arraySync();
    const predictedProbs = predictions.arraySync();
    const actualClasses = testLabels.arraySync();

    // Calculate metrics
    const metrics = this.calculateMetrics(actualClasses, predictedClasses);
    const cm = this.calculateConfusionMatrix(actualClasses, predictedClasses);

    const results = {
      accuracy: metrics.accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      f1_score: metrics.f1,
      confusion_matrix: cm,
      predictions: predictedClasses,
      prediction_probabilities: predictedProbs,
      class_report: this.generateClassificationReport(actualClasses, predictedClasses)
    };

    return results;
  }

  calculateMetrics(actual, predicted) {
    // Initialize counters for each class
    const classes = [0, 1, 2];
    const metrics = {
      tp: { 0: 0, 1: 0, 2: 0 },
      fp: { 0: 0, 1: 0, 2: 0 },
      fn: { 0: 0, 1: 0, 2: 0 },
      tn: { 0: 0, 1: 0, 2: 0 }
    };

    // Calculate TP, FP, FN, TN for each class
    for (let i = 0; i < actual.length; i++) {
      const a = actual[i];
      const p = predicted[i];
      
      for (const cls of classes) {
        if (a === cls && p === cls) metrics.tp[cls]++;
        if (a !== cls && p === cls) metrics.fp[cls]++;
        if (a === cls && p !== cls) metrics.fn[cls]++;
        if (a !== cls && p !== cls) metrics.tn[cls]++;
      }
    }

    // Calculate precision, recall, f1 for each class
    const precision = {};
    const recall = {};
    const f1 = {};

    for (const cls of classes) {
      precision[cls] = metrics.tp[cls] / (metrics.tp[cls] + metrics.fp[cls]) || 0;
      recall[cls] = metrics.tp[cls] / (metrics.tp[cls] + metrics.fn[cls]) || 0;
      f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) || 0;
    }

    // Calculate macro averages
    const macroPrecision = Object.values(precision).reduce((a, b) => a + b, 0) / classes.length;
    const macroRecall = Object.values(recall).reduce((a, b) => a + b, 0) / classes.length;
    const macroF1 = Object.values(f1).reduce((a, b) => a + b, 0) / classes.length;

    // Calculate accuracy
    const correct = actual.filter((a, i) => a === predicted[i]).length;
    const accuracy = correct / actual.length;

    return {
      accuracy,
      precision: { ...precision, macro: macroPrecision },
      recall: { ...recall, macro: macroRecall },
      f1: { ...f1, macro: macroF1 },
      metrics
    };
  }

  calculateConfusionMatrix(actual, predicted) {
    const cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // 3x3 matrix for 3 classes
    for (let i = 0; i < actual.length; i++) {
      const a = actual[i];
      const p = predicted[i];
      cm[a][p]++;
    }
    return cm;
  }

  generateClassificationReport(actual, predicted) {
    const metrics = this.calculateMetrics(actual, predicted);
    const report = {
      classes: this.classNames,
      accuracy: metrics.accuracy,
      macro_avg: {
        precision: metrics.precision.macro,
        recall: metrics.recall.macro,
        f1_score: metrics.f1.macro
      },
      class_metrics: {}
    };

    for (let i = 0; i < this.classNames.length; i++) {
      report.class_metrics[this.classNames[i]] = {
        precision: metrics.precision[i],
        recall: metrics.recall[i],
        f1_score: metrics.f1[i],
        support: actual.filter(val => val === i).length
      };
    }

    return report;
  }

  printEvaluationSummary(results) {
    console.log("\n" + "=".repeat(50));
    console.log("LUNGE STAGE MODEL EVALUATION");
    console.log("=".repeat(50));

    console.log(`Accuracy: ${results.accuracy.toFixed(4)}`);
    console.log(`Macro Precision: ${results.precision.macro.toFixed(4)}`);
    console.log(`Macro Recall: ${results.recall.macro.toFixed(4)}`);
    console.log(`Macro F1-Score: ${results.f1_score.macro.toFixed(4)}`);

    console.log("\nClass-wise Metrics:");
    console.log(`${'Class'.padEnd(10)} ${'Precision'.padEnd(12)} ${'Recall'.padEnd(12)} ${'F1-Score'.padEnd(12)} ${'Support'.padEnd(10)}`);
    console.log("-".repeat(60));
    
    for (const [cls, metrics] of Object.entries(results.class_report.class_metrics)) {
      console.log(
        `${cls.padEnd(10)} ${metrics.precision.toFixed(4).padEnd(12)} ${metrics.recall.toFixed(4).padEnd(12)} ` +
        `${metrics.f1_score.toFixed(4).padEnd(12)} ${metrics.support.toString().padEnd(10)}`
      );
    }

    console.log("\nConfusion Matrix:");
    console.log(`${'Actual \\ Pred'.padEnd(15)} ${this.classNames[0].padEnd(8)} ${this.classNames[1].padEnd(8)} ${this.classNames[2].padEnd(8)}`);
    console.log("-".repeat(40));
    for (let i = 0; i < this.classNames.length; i++) {
      console.log(
        `${this.classNames[i].padEnd(15)} ` +
        `${results.confusion_matrix[i][0].toString().padEnd(8)} ` +
        `${results.confusion_matrix[i][1].toString().padEnd(8)} ` +
        `${results.confusion_matrix[i][2].toString().padEnd(8)}`
      );
    }
  }

  async predict(features) {
    if (!this.isTrained) {
      throw new Error("Model must be trained before making predictions");
    }

    // Convert features to tensor if not already
    const featuresTensor = Array.isArray(features) ? 
      tf.tensor2d([features.map(val => parseFloat(val))]) :
      tf.tensor2d(features.map(row => Object.values(row).map(val => parseFloat(val))));

    // Scale features
    const scaledFeatures = featuresTensor.sub(this.scaler.mean).div(this.scaler.std);

    // Get probabilities and classes
    const probabilities = this.model.predict(scaledFeatures).arraySync();
    const classes = probabilities.map(probs => {
      const maxProb = Math.max(...probs);
      return probs.indexOf(maxProb);
    });

    return { 
      classes, 
      probabilities,
      classNames: classes.map(cls => this.classNames[cls])
    };
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
        mean: await this.scaler.mean.array(),
        std: await this.scaler.std.array()
      },
      featureNames: this.featureNames,
      labelMapping: this.labelMapping,
      classNames: this.classNames
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
      this.scaler = {
        mean: tf.tensor(modelData.scaler.mean),
        std: tf.tensor(modelData.scaler.std)
      };

      // Restore other properties
      this.featureNames = modelData.featureNames;
      this.labelMapping = modelData.labelMapping;
      this.classNames = modelData.classNames;
      this.isTrained = true;

      console.log(`Model loaded from ${filePath}`);
    } catch (error) {
      console.error(`Error loading model: ${error}`);
      throw error;
    }
  }

  async runCompletePipeline(trainFilePath, testFilePath = null, testSize = 0.2, modelParams = {}, saveModelPath = null) {
    console.log("Starting Complete ML Pipeline for Lunge Stage Detection");
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
        // FIX: Convert test labels to float32
        testLabels = tf.tensor1d(testProcessed.labels, 'float32');
        
        trainFeatures = tf.tensor2d(features.map(row => 
          Object.values(row).map(val => parseFloat(val)))
        );
        // FIX: Convert train labels to float32
        trainLabels = tf.tensor1d(labels, 'float32');
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

// Example usage
(async () => {
  try {
    const pipeline = new LungeStageMLPipeline(42);
    const results = await pipeline.runCompletePipeline(
      "./stage.train.csv",
      "./stage.test.csv",
      0.2,
      { 
        learningRate: 0.001, 
        epochs: 200,
        batchSize: 32,
        classWeight: { 0: 1, 1: 1.5, 2: 1.2 }
      },
      "./saved_models/lunge_stage_model"
    );
  } catch (error) {
    console.error("Error in pipeline:", error);
  }
})();