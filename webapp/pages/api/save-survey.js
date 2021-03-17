const { UserSession } = require('../../lib/data');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const sessionId = params.id;

  const session = await UserSession.findOne({ id: sessionId });
  session.survey = params.survey;
  session.endTimestamp = new Date();
  await session.save();

  res.json({ mechanicalTurkersAreAwesome: true });
}
