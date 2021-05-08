const { UserSession } = require('../../lib/data');
const _ = require('lodash');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const sessionId = params.id;
  const qStr = params.qStr;
  const qAns = params.qAns;
  const timestamp = _.now();

  if (!(sessionId)) {

    return res.json({ "error": "No session ID provided." });
  }
  const session = await UserSession.findOne({ id: sessionId });
  session.exerciseResponses = _.concat(session.exerciseResponses, {"question": qStr, "answer": qAns, "timestamp": timestamp});
  console.log(session, qAns)
  await session.save();

  res.json({});
};
