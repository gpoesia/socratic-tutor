const { UserSession } = require('../../lib/data');
const uuid = require('uuid');

export default async (req, res) => {
  const newSessionId = uuid.v4();
  const params = JSON.parse(req.query.params || '{}');

  const session = new UserSession({ id: newSessionId });
  const curriculum = params.curriculum || 'random';

  console.log('Curriculum:', curriculum);

  session.beginTimestamp = new Date();
  session.preTestResponses = [];
  session.exerciseResponses = [];
  session.curriculum = curriculum;
  session.postTestResponses = [];

  await session.save();

  res.send({ id: newSessionId });
}
