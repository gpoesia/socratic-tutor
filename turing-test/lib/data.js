const Mongoose = require('mongoose');
const Config = require('../config.json');

Mongoose.connect(`mongodb://${Config.mongodbHost}/tutor`,
                 { useNewUrlParser: true, useUnifiedTopology: true });

module.exports.UserSession = Mongoose.models.UserSession || Mongoose.model(
  'UserSession',
  new Mongoose.Schema({
    id: String,
    type: String,
    beginTimestamp: Date,
    endTimestamp: Date,
    exerciseResponses: Array,
    survey: Object,
  }),
);