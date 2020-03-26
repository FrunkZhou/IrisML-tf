module.exports = function(app) {
  var irisML = require("../controllers/irisML.server.controller");

  app.route("/").get(irisML.getData);

  app
    .route("/analyze")
    .post(irisML.uploadDataset, irisML.prepareModel, irisML.renderIrisForm);

  app.route("/results").post(irisML.predict);
};
