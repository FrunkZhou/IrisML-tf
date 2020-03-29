const formidable = require("formidable");
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

var tfModel;
var dataset;

exports.getData = (req, res) => {
  res.render("dataUpload");
};

exports.uploadDataset = (req, res, next) => {
  var form = new formidable.IncomingForm();

  //OS TMP dir by default
  form.uploadDir = "./app/uploads";

  //Rename file to original filename
  form.on("file", (field, file) => {
    fs.rename(file.path, form.uploadDir + "/" + file.name, err => {
      if (err) {
        console.log(err);
      }
    });
    req.filename = file.name;
  });

  form.parse(req, (err, fields, files) => {
    if (err) {
      console.log(err);
    }
    next();
  });
};

exports.prepareModel = (req, res, next) => {
  var filePath = path.join(__dirname, "../uploads/" + req.filename);
  fs.readFile(filePath, (err, data) => {
    //Provided file is utf-8 BOM, we need to remove endianess
    dataset = JSON.parse(data.toString("utf8").replace(/^\uFEFF/, ""));
    tfModel = prepareTFModel();
  });
  next();
};

var prepareTFModel = () => {
  // we use a sequential neural network
  const model = tf.sequential();
  //add the first layer
  model.add(
    tf.layers.dense({
      inputShape: [4], // four features
      activation: "sigmoid",
      units: 5 // five hidden layers
    })
  );
  //add the hidden layer
  model.add(
    tf.layers.dense({
      inputShape: [5],
      activation: "sigmoid",
      units: 3 //3 output categorizations
    })
  );
  //add output layer
  model.add(
    tf.layers.dense({
      activation: "sigmoid",
      units: 3 //dimension of final output
    })
  );
  return model;
};

exports.renderIrisForm = (req, res) => {
  res.render("irisTestForm");
};

exports.predict = (req, res) => {
  // TODO: make testset simply accept user form data instead of forming it back into an array
  // this is such a bad way to do it, need to fix
  var testset = [
    {
      // Coerce the values to numbers, since form sets them as strings
      sepal_length: +req.body.sepalLength,
      sepal_width: +req.body.sepalWidth,
      petal_length: +req.body.petalLength,
      petal_width: +req.body.petalWidth
    }
  ];
  getPrediction(
    tfModel,
    { training: dataset, test: testset },
    {
      learningEpoch: req.body.learningEpoch,
      learningRate: req.body.learningRate
    },
    data => {
      res.render("results", { data });
    }
  );
};

var getPrediction = (model, data, parameters, callback) => {
  //data features
  const trainingData = tf.tensor2d(
    data.training.map(item => [
      item.sepal_length,
      item.sepal_width,
      item.petal_length,
      item.petal_width
    ])
  );
  //data output as dichotomous values
  const outputData = tf.tensor2d(
    data.training.map(item => [
      item.species === "setosa" ? 1 : 0,
      item.species === "virginica" ? 1 : 0,
      item.species === "versicolor" ? 1 : 0
    ])
  );

  const outputMap = ["setosa", "virginica", "versicolor"];

  //
  //testing data features
  const testingData = tf.tensor2d(
    data.test.map(item => [
      item.sepal_length,
      item.sepal_width,
      item.petal_length,
      item.petal_width
    ])
  );

  //compile the model with an MSE loss function and Adam algorithm
  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(parameters.learningRate)
  });

  model
    .fit(trainingData, outputData, { epochs: parameters.learningEpoch })
    .then(history => {
      var predictResult = model.predict(testingData).dataSync();
      var payload = {
        predictOdds: (Math.max(...predictResult) * 100).toFixed(2),
        species: outputMap[predictResult.indexOf(Math.max(...predictResult))]
      };
      callback(payload);
    });
};
