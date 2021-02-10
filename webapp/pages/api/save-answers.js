const { UserSession } = require('../../lib/data');
const _ = require('lodash');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const sessionId = params.id;
  const stage = params.stage;
  const timestamp = _.now();
  const answers = params.answers.map(a => ({ ...a, timestamp }));

  if (!(sessionId && stage)) {
    return res.json({ "error": "No session ID or stage provided." });
  }

  const session = await UserSession.findOne({ id: sessionId });

  if (stage === "assessment-pre") {
    session.preTestResponses = _.concat(session.preTestResponses, answers);
  } else if (stage === "assessment-post") {
    session.postTestResponses = _.concat(session.postTestResponses, answers);
  } else if (stage === "exercises") {
    session.exerciseResponses = _.concat(session.exerciseResponses, answers);
  } else {
    return res.json({ "error": "Invalid stage " + stage });
  }

  await session.save();
  res.json({});
};
