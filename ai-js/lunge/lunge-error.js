const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

class LungeErrorMLPipeline {
  constructor(randomState = 42) {
    this.randomState = randomState;
    this.model = null;
    this.scaler = null;
    this.isTrained = false;
    this.featureNames = null;
    this.labelMapping = { 'correct': 0, 'knee_collapse': 1, 'back_bend': 2, 'shallow': 3 };
    this.classNames = ['correct', 'knee_collapse', 'back_bend', 'shallow'];
    this.bestWeights = null;
    this.bestValLoss = Infinity;
    this.patience = 20;
    this.wait = 0;
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

    // Convert string labels to numeric - FIX: Handle undefined labels
    processedData.forEach(row => {
      if (row.label === undefined || row.label === 'undefined') {
        // If label is undefined, set to 0 (correct) as default
        row.label = 0;
      } else {
        row.label = this.labelMapping[row.label] !== undefined ? this.labelMapping[row.label] : 0;
      }
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
      const labelName = this.classNames[label] || 'unknown';
      acc[labelName] = (acc[labelName] || 0) + 1;
      return acc;
    }, {});
    
    console.log(`Labels distribution: ${JSON.stringify(labelCounts)}`);

    return { features, labels };
  }

  async splitData(features, labels, testSize = 0.2) {
    // Convert to tensors with proper types
    const featuresTensor = tf.tensor2d(
      features.map(row => 
        Object.values(row).map(val => parseFloat(val))
      ),
      [features.length, Object.keys(features[0]).length],
      'float32'
    );
    
    // FIX: Create labels tensor as float32 instead of int32
    const labelsTensor = tf.tensor1d(labels.map(label => parseFloat(label)), 'float32');

    // Calculate split index
    const splitIdx = Math.floor(featuresTensor.shape[0] * (1 - testSize));

    // Split the data
    const trainFeatures = featuresTensor.slice(0, splitIdx);
    const testFeatures = featuresTensor.slice(splitIdx);
    const trainLabels = labelsTensor.slice(0, splitIdx);
    const testLabels = labelsTensor.slice(splitIdx);

    console.log(`Training set size: ${trainFeatures.shape[0]}`);
    console.log(`Test set size: ${testFeatures.shape[0]}`);

    // Dispose intermediate tensors
    featuresTensor.dispose();
    labelsTensor.dispose();

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
    
    // Handle division by zero
    const safeStd = tf.where(
      tf.equal(std, 0),
      tf.onesLike(std),
      std
    );
    
    this.scaler = { 
      mean: mean, 
      std: safeStd 
    };

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
      classWeight: { 0: 1, 1: 2, 2: 2, 3: 1.5 }
    };
    
    const params = { ...defaultParams, ...modelParams };

    // Build the model
    this.model = tf.sequential();
    
    // Input layer
    this.model.add(tf.layers.dense({
      units: 256,
      inputShape: [trainFeatures.shape[1]],
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));
    
    // Hidden layers
    this.model.add(tf.layers.dropout({ rate: 0.4 }));
    this.model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));
    
    this.model.add(tf.layers.dropout({ rate: 0.3 }));
    this.model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));
    
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({
      units: 32,
      activation: 'relu'
    }));
    
    // Output layer
    this.model.add(tf.layers.dense({
      units: 4,
      activation: 'softmax'
    }));

    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(params.learningRate),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });

    // Custom training loop with early stopping
    const batchSize = params.batchSize;
    const validationSplit = params.validationSplit;
    const numTrainSamples = trainFeatures.shape[0];
    const numValidationSamples = Math.floor(numTrainSamples * validationSplit);
    const numBatchesPerEpoch = Math.ceil(numTrainSamples * (1 - validationSplit) / batchSize);

    // Split into training and validation sets
    const trainX = trainFeatures.slice(0, numTrainSamples - numValidationSamples);
    const trainY = trainLabels.slice(0, numTrainSamples - numValidationSamples);
    const valX = trainFeatures.slice(numTrainSamples - numValidationSamples);
    const valY = trainLabels.slice(numTrainSamples - numValidationSamples);

    for (let epoch = 0; epoch < params.epochs; epoch++) {
      console.log(`Epoch ${epoch + 1}/${params.epochs}`);
      
      // Train on batches
      let epochLoss = 0;
      for (let batch = 0; batch < numBatchesPerEpoch; batch++) {
        const batchStart = batch * batchSize;
        const batchEnd = Math.min(batchStart + batchSize, trainX.shape[0]);
        
        const xBatch = trainX.slice(batchStart, batchEnd - batchStart);
        const yBatch = trainY.slice(batchStart, batchEnd - batchStart);
        
        const history = await this.model.trainOnBatch(xBatch, yBatch);
        epochLoss += history[0];
        
        // Clean up
        xBatch.dispose();
        yBatch.dispose();
      }
      
      // Validate
      const valOut = this.model.evaluate(valX, valY, { batchSize: batchSize });
      const valLoss = await valOut[0].data();
      const valAcc = await valOut[1].data();
      
      console.log(`loss: ${(epochLoss / numBatchesPerEpoch).toFixed(4)} - ` +
                  `val_loss: ${valLoss[0].toFixed(4)} - val_acc: ${valAcc[0].toFixed(4)}`);
      
      // Early stopping and best weights tracking
      if (valLoss[0] < this.bestValLoss) {
        this.bestValLoss = valLoss[0];
        this.bestWeights = await this.model.getWeights();
        this.wait = 0;
        console.log(`New best validation loss: ${this.bestValLoss.toFixed(4)}`);
      } else {
        this.wait++;
        if (this.wait >= this.patience) {
          console.log(`Early stopping at epoch ${epoch + 1}`);
          break;
        }
      }
      
      // Clean up
      valOut[0].dispose();
      valOut[1].dispose();
    }

    // Restore best weights
    if (this.bestWeights) {
      console.log('Restoring best weights...');
      await this.model.setWeights(this.bestWeights);
    }

    this.isTrained = true;
    console.log("Model trained successfully");
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
    const classes = [0, 1, 2, 3];
    const metrics = {
      tp: { 0: 0, 1: 0, 2: 0, 3: 0 },
      fp: { 0: 0, 1: 0, 2: 0, 3: 0 },
      fn: { 0: 0, 1: 0, 2: 0, 3: 0 },
      tn: { 0: 0, 1: 0, 2: 0, 3: 0 }
    };

    for (let i = 0; i < actual.length; i++) {
      const a = Math.floor(actual[i]); // FIX: Ensure integer values
      const p = predicted[i];
      
      for (const cls of classes) {
        if (a === cls && p === cls) metrics.tp[cls]++;
        if (a !== cls && p === cls) metrics.fp[cls]++;
        if (a === cls && p !== cls) metrics.fn[cls]++;
        if (a !== cls && p !== cls) metrics.tn[cls]++;
      }
    }

    const precision = {};
    const recall = {};
    const f1 = {};

    for (const cls of classes) {
      precision[cls] = metrics.tp[cls] / (metrics.tp[cls] + metrics.fp[cls]) || 0;
      recall[cls] = metrics.tp[cls] / (metrics.tp[cls] + metrics.fn[cls]) || 0;
      f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) || 0;
    }

    const macroPrecision = Object.values(precision).reduce((a, b) => a + b, 0) / classes.length;
    const macroRecall = Object.values(recall).reduce((a, b) => a + b, 0) / classes.length;
    const macroF1 = Object.values(f1).reduce((a, b) => a + b, 0) / classes.length;

    const correct = actual.filter((a, i) => Math.floor(a) === predicted[i]).length;
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
    const cm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    for (let i = 0; i < actual.length; i++) {
      const a = Math.floor(actual[i]); // FIX: Ensure integer values
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
        support: actual.filter(val => Math.floor(val) === i).length
      };
    }

    return report;
  }

  printEvaluationSummary(results) {
    console.log("\n" + "=".repeat(50));
    console.log("LUNGE ERROR MODEL EVALUATION");
    console.log("=".repeat(50));

    console.log(`Accuracy: ${results.accuracy.toFixed(4)}`);
    console.log(`Macro Precision: ${results.precision.macro.toFixed(4)}`);
    console.log(`Macro Recall: ${results.recall.macro.toFixed(4)}`);
    console.log(`Macro F1-Score: ${results.f1_score.macro.toFixed(4)}`);

    console.log("\nClass-wise Metrics:");
    console.log(`${'Class'.padEnd(15)} ${'Precision'.padEnd(12)} ${'Recall'.padEnd(12)} ${'F1-Score'.padEnd(12)} ${'Support'.padEnd(10)}`);
    console.log("-".repeat(65));
    
    for (const [cls, metrics] of Object.entries(results.class_report.class_metrics)) {
      console.log(
        `${cls.padEnd(15)} ${metrics.precision.toFixed(4).padEnd(12)} ${metrics.recall.toFixed(4).padEnd(12)} ` +
        `${metrics.f1_score.toFixed(4).padEnd(12)} ${metrics.support.toString().padEnd(10)}`
      );
    }

    console.log("\nConfusion Matrix:");
    console.log(`${'Actual \\ Pred'.padEnd(15)} ${this.classNames[0].padEnd(8)} ${this.classNames[1].padEnd(8)} ${this.classNames[2].padEnd(8)} ${this.classNames[3].padEnd(8)}`);
    console.log("-".repeat(55));
    for (let i = 0; i < this.classNames.length; i++) {
      console.log(
        `${this.classNames[i].padEnd(15)} ` +
        `${results.confusion_matrix[i][0].toString().padEnd(8)} ` +
        `${results.confusion_matrix[i][1].toString().padEnd(8)} ` +
        `${results.confusion_matrix[i][2].toString().padEnd(8)} ` +
        `${results.confusion_matrix[i][3].toString().padEnd(8)}`
      );
    }
  }

  async predict(features) {
    if (!this.isTrained) {
      throw new Error("Model must be trained before making predictions");
    }

    // Convert features to tensor if not already
    const featuresTensor = Array.isArray(features) ? 
      tf.tensor2d([features.map(val => parseFloat(val))], [1, features.length], 'float32') :
      tf.tensor2d(features.map(row => Object.values(row).map(val => parseFloat(val))), 
        [features.length, this.featureNames.length], 
        'float32'
      );

    // Scale features
    const scaledFeatures = featuresTensor.sub(this.scaler.mean).div(this.scaler.std);

    // Get probabilities and classes
    const probabilities = this.model.predict(scaledFeatures).arraySync();
    const classes = probabilities.map(probs => {
      const maxProb = Math.max(...probs);
      return probs.indexOf(maxProb);
    });

    // Clean up
    featuresTensor.dispose();
    scaledFeatures.dispose();

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
    console.log("Starting Complete ML Pipeline for Lunge Error Detection");
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
        testFeatures = tf.tensor2d(
          testProcessed.features.map(row => 
            Object.values(row).map(val => parseFloat(val))
          ),
          [testProcessed.features.length, this.featureNames.length],
          'float32'
        );
        // FIX: Create test labels as float32
        testLabels = tf.tensor1d(testProcessed.labels.map(label => parseFloat(label)), 'float32');
        
        trainFeatures = tf.tensor2d(
          features.map(row => 
            Object.values(row).map(val => parseFloat(val))
          ),
          [features.length, this.featureNames.length],
          'float32'
        );
        // FIX: Create train labels as float32
        trainLabels = tf.tensor1d(labels.map(label => parseFloat(label)), 'float32');
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
    } finally {
      // Clean up any remaining tensors
      tf.disposeVariables();
    }
  }
}

// Example usage
(async () => {
  try {
    // Initialize pipeline
    const pipeline = new LungeErrorMLPipeline(42);

    // Run complete pipeline
    const results = await pipeline.runCompletePipeline(
      "./err.train.csv",
      "./err.test.csv",
      0.2,
      { 
        learningRate: 0.001, 
        epochs: 200,
        batchSize: 32,
        classWeight: { 0: 1, 1: 2, 2: 2, 3: 1.5 }
      },
      "./saved_models/lunge_error_model"
    );

  } catch (error) {
    console.error("Error in pipeline:", error);
  }
})();